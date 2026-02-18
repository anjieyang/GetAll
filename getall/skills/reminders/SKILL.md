---
name: reminders
description: "Schedule reminders and recurring tasks. Use when user says '提醒我', 'remind me', 'every X minutes', '定时', 'schedule', or wants to set up periodic reports and alerts."
---

# Reminders

Use the `reminders` tool to schedule reminders or recurring tasks.

If the user gives a clear scheduling instruction, create it immediately.
Do not ask for a second confirmation like "reply start".

Use `mode`:

- `mode: "direct"` (default): send message text as-is when triggered.
- `mode: "agent"`: run an agent turn on each trigger (for periodic analysis/reporting).

## Three Modes

1. **Reminder** - message is sent directly to user
2. **Task** - message is a task description, agent executes and sends result
3. **One-time** - runs once at a specific time, then auto-deletes

## Examples

Fixed reminder:

```text
reminders(action="add", message="Time to take a break!", every_seconds=1200)
```

Dynamic task (agent executes each time):

```text
reminders(action="add", message="Check Bitget/GetAll GitHub stars and report", every_seconds=600)

# explicitly run analysis each trigger
reminders(action="add", message="Generate a daily BTC/ETH report", cron_expr="0 9 * * *", mode="agent")
```

Finite sequence (auto-stop):

```text
reminders(action="add", message="Send next number in sequence", every_seconds=1, max_runs=10, final_message="Done.")
```

One-time scheduled task (compute ISO datetime from current time):

```text
reminders(action="add", message="Remind me about the meeting", at="<ISO datetime>")
```

List/remove:

```text
reminders(action="list")
reminders(action="remove", job_id="abc123")
```

## Task-Scoped Heartbeat

For long-running operations (backtest, large data fetch), use task-scoped heartbeats
to keep the user informed. These auto-cancel when the current task completes.

```text
# Before calling a long tool, schedule a progress check-in
reminders(task_scoped=true, delay_seconds=60, message="回测还在跑，再等一会儿~")

# Then call the long-running tool
run_backtest(...)
```

**Rules:**
- Only set heartbeats for tasks you expect to take >30 seconds
- Use "Recent Tool Execution Times" in context to estimate duration
- Set delay_seconds to ~50-70% of expected duration
- Keep messages natural and short, especially in group chats
- Heartbeats auto-cancel — no cleanup needed

## Time Expressions

| User says          | Parameters                                          |
| ------------------ | --------------------------------------------------- |
| every 20 minutes   | every_seconds: 1200                                 |
| every hour         | every_seconds: 3600                                 |
| every day at 8am   | cron_expr: `"0 8 * * *"`                            |
| weekdays at 5pm    | cron_expr: `"0 17 * * 1-5"`                         |
| at a specific time | at: ISO datetime string (compute from current time) |
