# Heartbeat Tasks

Checked every ~30 minutes. Batch all checks in one pass — only speak up when something needs attention.

> **Cron jobs** (anomaly-scan, position-sync, news-feed, daily-review) run independently. Use `reminders` tool to manage them.
> **WebSocket services** (ws:account, ws:breaking-news) are autonomous. Use `service` tool to manage them.

## Periodic Awareness

- Glance at `memory/trading/anomalies.jsonl` — any unresolved HIGH severity anomalies?
- Check cron job health (`reminders(action="list")`) — any consecutive failures?
- Review `memory/trading/positions.json` — any positions approaching liquidation or stop-loss?
- If user left unfinished tasks in recent conversation → follow up

## Ad-hoc Tasks

<!-- User or agent can add temporary tasks below -->

