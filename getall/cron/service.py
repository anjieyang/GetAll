"""Cron service for scheduling agent tasks.

Enhanced with retry, alerting, stuck-job detection, and missed-job catch-up.
"""

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from getall.cron.types import CronJob, CronJobConfig, CronJobState, CronPayload, CronSchedule, CronStore

# Fallback stuck-job timeout (2 h) – overridden by job.config.timeout_ms
_DEFAULT_STUCK_TIMEOUT_MS = 2 * 60 * 60 * 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_job_name(name: str) -> str:
    """Normalize for idempotent comparison (lowercase, underscores)."""
    return re.sub(r"[-\s]+", "_", name.lower().strip())


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        if not schedule.at_ms:
            return None
        return schedule.at_ms if schedule.at_ms > now_ms else now_ms + 500

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from datetime import datetime as _dt
            from croniter import croniter

            if schedule.tz:
                try:
                    import pytz
                    tz = pytz.timezone(schedule.tz)
                    start_time = _dt.now(tz)
                except Exception:
                    start_time = _dt.now()
            else:
                start_time = _dt.now()

            cron = croniter(schedule.expr, start_time)
            next_time = cron.get_next(_dt)
            return int(next_time.timestamp() * 1000)
        except Exception:
            return None

    return None


class CronService:
    """Manages and executes scheduled jobs with retry / alerting / stuck-detection."""

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        on_error: Callable[[str], Coroutine[Any, Any, str | None]] | None = None,
    ):
        self.store_path = store_path
        self.on_job = on_job
        self.on_error = on_error
        self._store: CronStore | None = None
        self._timer_task: asyncio.Task | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_store(self) -> CronStore:
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                jobs: list[CronJob] = []
                for j in data.get("jobs", []):
                    jobs.append(CronJob(
                        id=j["id"],
                        name=j["name"],
                        enabled=j.get("enabled", True),
                        schedule=CronSchedule(
                            kind=j["schedule"]["kind"],
                            at_ms=j["schedule"].get("atMs"),
                            every_ms=j["schedule"].get("everyMs"),
                            expr=j["schedule"].get("expr"),
                            tz=j["schedule"].get("tz"),
                        ),
                        payload=CronPayload(
                            kind=j["payload"].get("kind", "direct_message"),
                            message=j["payload"].get("message", ""),
                            deliver=j["payload"].get("deliver", False),
                            channel=j["payload"].get("channel"),
                            to=j["payload"].get("to"),
                            final_message=j["payload"].get("finalMessage"),
                            tenant_id=j["payload"].get("tenantId", "default"),
                            principal_id=j["payload"].get("principalId", ""),
                            agent_identity_id=j["payload"].get("agentIdentityId", ""),
                            sender_id=j["payload"].get("senderId", ""),
                            thread_id=j["payload"].get("threadId", ""),
                            chat_type=j["payload"].get("chatType", "private"),
                        ),
                        state=CronJobState(
                            next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                            last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                            run_count=j.get("state", {}).get("runCount", 0),
                            last_status=j.get("state", {}).get("lastStatus"),
                            last_error=j.get("state", {}).get("lastError"),
                            consecutive_failures=j.get("state", {}).get("consecutiveFailures", 0),
                            running_at_ms=j.get("state", {}).get("runningAtMs"),
                            retry_count=j.get("state", {}).get("retryCount", 0),
                        ),
                        config=CronJobConfig(
                            timeout_ms=j.get("config", {}).get("timeoutMs", 7_200_000),
                            max_retries=j.get("config", {}).get("maxRetries", 3),
                            retry_backoff_base=j.get("config", {}).get("retryBackoffBase", 2.0),
                            alert_threshold=j.get("config", {}).get("alertThreshold", 5),
                        ),
                        created_at_ms=j.get("createdAtMs", 0),
                        updated_at_ms=j.get("updatedAtMs", 0),
                        delete_after_run=j.get("deleteAfterRun", False),
                        max_runs=j.get("maxRuns"),
                    ))
                self._store = CronStore(jobs=jobs)
            except Exception as exc:
                logger.warning(f"Failed to load cron store: {exc}")
                self._store = CronStore()
        else:
            self._store = CronStore()

        return self._store

    def _save_store(self) -> None:
        if not self._store:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self._store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                        "finalMessage": j.payload.final_message,
                        "tenantId": j.payload.tenant_id,
                        "principalId": j.payload.principal_id,
                        "agentIdentityId": j.payload.agent_identity_id,
                        "senderId": j.payload.sender_id,
                        "threadId": j.payload.thread_id,
                        "chatType": j.payload.chat_type,
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "runCount": j.state.run_count,
                        "lastStatus": j.state.last_status,
                        "lastError": j.state.last_error,
                        "consecutiveFailures": j.state.consecutive_failures,
                        "runningAtMs": j.state.running_at_ms,
                        "retryCount": j.state.retry_count,
                    },
                    "config": {
                        "timeoutMs": j.config.timeout_ms,
                        "maxRetries": j.config.max_retries,
                        "retryBackoffBase": j.config.retry_backoff_base,
                        "alertThreshold": j.config.alert_threshold,
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                    "maxRuns": j.max_runs,
                }
                for j in self._store.jobs
            ],
        }
        self.store_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._load_store()
        self._clear_stuck_jobs()
        self._gc()
        self._recompute_next_runs()
        self._save_store()
        await self._run_missed_jobs()
        self._arm_timer()
        logger.info(f"Cron service started with {len(self._store.jobs if self._store else [])} jobs")

    def stop(self) -> None:
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    # ------------------------------------------------------------------
    # Resilience helpers
    # ------------------------------------------------------------------

    def _clear_stuck_jobs(self) -> None:
        """Reset stale ``running_at_ms`` markers (stuck > configured timeout)."""
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if job.state.running_at_ms:
                timeout = job.config.timeout_ms if job.config else _DEFAULT_STUCK_TIMEOUT_MS
                if (now - job.state.running_at_ms) > timeout:
                    logger.warning(f"Cron: clearing stuck job '{job.name}'")
                    job.state.running_at_ms = None
                    job.state.last_status = "error"
                    job.state.last_error = "Job stuck - cleared on startup"
                    job.state.retry_count = 0

    async def _run_missed_jobs(self) -> None:
        """Catch-up jobs that were due while the service was stopped."""
        if not self._store:
            return
        now = _now_ms()
        missed = [
            j for j in self._store.jobs
            if j.enabled
            and j.state.next_run_at_ms
            and j.state.next_run_at_ms <= now
            and j.schedule.kind != "at"
        ]
        if missed:
            logger.info(f"Cron: {len(missed)} missed jobs detected, running catch-up")
            for job in missed:
                await self._execute_job(job)
            self._save_store()

    def _gc(self) -> None:
        """Remove dead jobs."""
        if not self._store:
            return
        now = _now_ms()
        before = len(self._store.jobs)
        alive: list[CronJob] = []
        for j in self._store.jobs:
            if j.schedule.kind == "at" and not j.enabled:
                continue
            if (
                j.schedule.kind == "at"
                and j.schedule.at_ms
                and j.schedule.at_ms < now - 60_000
                and j.state.last_run_at_ms is None
            ):
                continue
            if j.max_runs is not None and j.state.run_count >= j.max_runs:
                continue
            alive.append(j)
        self._store.jobs = alive
        removed = before - len(alive)
        if removed:
            logger.info(f"Cron GC: removed {removed} dead jobs, {len(alive)} remain")

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _recompute_next_runs(self) -> None:
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if job.enabled:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    def _get_next_wake_ms(self) -> int | None:
        if not self._store:
            return None
        times = [j.state.next_run_at_ms for j in self._store.jobs if j.enabled and j.state.next_run_at_ms]
        return min(times) if times else None

    def _arm_timer(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()
        next_wake = self._get_next_wake_ms()
        if not next_wake or not self._running:
            return
        delay_s = max(0, next_wake - _now_ms()) / 1000

        async def tick() -> None:
            await asyncio.sleep(delay_s)
            if self._running:
                await self._on_timer()

        self._timer_task = asyncio.create_task(tick())

    async def _on_timer(self) -> None:
        if not self._store:
            return
        now = _now_ms()
        due = [j for j in self._store.jobs if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms]
        for job in due:
            await self._execute_job(job)
        self._gc()
        self._save_store()
        self._arm_timer()

    # ------------------------------------------------------------------
    # Execution (with retry + alerting)
    # ------------------------------------------------------------------

    async def _execute_job(self, job: CronJob) -> None:
        start_ms = _now_ms()
        job.state.running_at_ms = start_ms
        job.state.run_count += 1
        logger.info(f"Cron: executing '{job.name}' ({job.id})")

        try:
            if self.on_job:
                await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            job.state.consecutive_failures = 0
            job.state.retry_count = 0
            logger.info(f"Cron: '{job.name}' completed")

        except Exception as exc:
            job.state.last_status = "error"
            job.state.last_error = str(exc)
            job.state.consecutive_failures += 1
            job.state.retry_count += 1

            cfg = job.config or CronJobConfig()
            logger.error(
                f"Cron: '{job.name}' failed "
                f"(attempt {job.state.retry_count}/{cfg.max_retries}, "
                f"consecutive: {job.state.consecutive_failures}): {exc}"
            )

            # Retry with exponential backoff
            if job.state.retry_count < cfg.max_retries:
                backoff = cfg.retry_backoff_base ** (job.state.retry_count - 1)
                logger.info(f"Cron: retrying '{job.name}' in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                if self._running:
                    await self._execute_job(job)
                    return
            else:
                job.state.retry_count = 0
                # Alert on prolonged failure
                if job.state.consecutive_failures >= cfg.alert_threshold and self.on_error:
                    alert = (
                        f"Cron Job Alert: '{job.name}' has failed "
                        f"{job.state.consecutive_failures} times consecutively. "
                        f"Last error: {exc}"
                    )
                    logger.critical(alert)
                    try:
                        await self.on_error(alert)
                    except Exception as ae:
                        logger.warning(f"Cron: alert callback failed: {ae}")

                # Push every failure to agent for awareness
                if self.on_error:
                    try:
                        await self.on_error(
                            f"Cron job '{job.name}' failed "
                            f"(consecutive: {job.state.consecutive_failures}). "
                            f"Error: {exc}"
                        )
                    except Exception:
                        pass

        job.state.running_at_ms = None
        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = _now_ms()

        # Finite-run guard
        if job.max_runs is not None and job.state.run_count >= job.max_runs:
            self._store.jobs = [j for j in self._store.jobs if j.id != job.id]  # type: ignore[union-attr]
            logger.info(f"Cron: '{job.name}' reached max_runs={job.max_runs}, removed")
            return

        # One-shot handling
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]  # type: ignore[union-attr]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

    # ==================================================================
    # Public API
    # ==================================================================

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        payload_kind: str = "agent_turn",
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        max_runs: int | None = None,
        final_message: str | None = None,
        tenant_id: str = "default",
        principal_id: str = "",
        agent_identity_id: str = "",
        sender_id: str = "",
        thread_id: str = "",
        chat_type: str = "private",
    ) -> CronJob:
        """Add a job (idempotent – updates existing job with same normalized name)."""
        store = self._load_store()
        now = _now_ms()
        normalized = _normalize_job_name(name)

        # Idempotent: update existing job with same name
        for existing in store.jobs:
            if _normalize_job_name(existing.name) == normalized:
                existing.name = name
                existing.schedule = schedule
                existing.payload.message = message
                existing.payload.deliver = deliver
                existing.payload.channel = channel
                existing.payload.to = to
                existing.payload.final_message = final_message
                existing.enabled = True
                existing.updated_at_ms = now
                existing.state.next_run_at_ms = _compute_next_run(schedule, now)
                existing.delete_after_run = delete_after_run
                if max_runs is not None and max_runs > 0:
                    existing.max_runs = max_runs
                self._save_store()
                self._arm_timer()
                logger.info(f"Cron: updated existing '{name}' ({existing.id})")
                return existing

        normalized_max_runs = max_runs if isinstance(max_runs, int) and max_runs > 0 else None
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind=payload_kind if payload_kind in {"agent_turn", "direct_message", "system_event"} else "agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
                final_message=final_message,
                tenant_id=tenant_id or "default",
                principal_id=principal_id or "",
                agent_identity_id=agent_identity_id or "",
                sender_id=sender_id or "",
                thread_id=thread_id or "",
                chat_type=chat_type or "private",
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
            max_runs=normalized_max_runs,
        )
        store.jobs.append(job)
        self._save_store()
        self._arm_timer()
        logger.info(f"Cron: added '{name}' ({job.id})")
        return job

    def remove_job(self, job_id: str) -> bool:
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != job_id]
        if len(store.jobs) < before:
            self._save_store()
            self._arm_timer()
            logger.info(f"Cron: removed job {job_id}")
            return True
        return False

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                job.enabled = enabled
                job.updated_at_ms = _now_ms()
                job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms()) if enabled else None
                self._save_store()
                self._arm_timer()
                return job
        return None

    def update_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        message: str | None = None,
        every_ms: int | None = None,
        cron_expr: str | None = None,
        timezone: str | None = None,
        channel: str | None = None,
        to: str | None = None,
    ) -> CronJob | None:
        """Partial update – only non-``None`` params are modified."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if name is not None:
                    job.name = name
                if message is not None:
                    job.payload.message = message
                if channel is not None:
                    job.payload.channel = channel
                if to is not None:
                    job.payload.to = to

                schedule_changed = False
                if every_ms is not None:
                    job.schedule = CronSchedule(kind="every", every_ms=every_ms)
                    schedule_changed = True
                elif cron_expr is not None:
                    tz = timezone or job.schedule.tz or "Asia/Shanghai"
                    job.schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
                    schedule_changed = True
                elif timezone is not None and job.schedule.kind == "cron":
                    job.schedule.tz = timezone
                    schedule_changed = True

                job.updated_at_ms = _now_ms()
                if job.enabled and schedule_changed:
                    job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                self._save_store()
                self._arm_timer()
                logger.info(f"Cron: updated '{job.name}' ({job.id})")
                return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if not force and not job.enabled:
                    return False
                await self._execute_job(job)
                self._save_store()
                self._arm_timer()
                return True
        return False

    def status(self) -> dict:
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
