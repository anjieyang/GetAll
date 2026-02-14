---
name: meme-hunter
description: "Search the web for memes/reaction images or GIFs, download locally, and send inline in chat. Use when user asks for 表情包, 梗图, 斗图, reaction image, meme, or GIF."
metadata: {"getall":{"emoji":"😂","requires":{"bins":["python3.11"]}}}
---

# Meme Hunter

Use this skill when user asks for:
- "表情包", "梗图", "斗图"
- reaction image / reaction GIF / meme image
- funny image for current conversation mood

Goal: do everything end-to-end yourself (search -> download -> send). Do not ask user to provide image URLs.

## Workflow

1. Decide a concise search query from conversation context.
2. Run the bundled script to auto-search and download one image:

```bash
python3.11 getall/skills/meme-hunter/scripts/search_meme.py --query "happy reaction meme" --prefer-gif
```

3. Parse JSON output:
   - success: `{"ok": true, "path": "...", ...}`
   - failure: `{"ok": false, "error": "..."}`
4. On success, send a short natural line plus the local file path in your reply.
   - The platform will auto-upload and render the image from that path.
5. If first query fails, retry with 1-2 alternative phrasings before giving up.

## Good Query Strategy

- Keep it short and visual-first, e.g.:
  - `开心 表情包`
  - `无语 reaction meme`
  - `bull market celebration gif`
  - `社死 尴尬 meme`
- Prefer emotion + scene, not long sentences.

## Safety and Quality

- Avoid NSFW, violent, hateful, or illegal content.
- Prefer clean, low-noise images suitable for group chats.
- If user asks for multiple options, run script multiple times with different queries.

## Fallback (when script repeatedly fails)

1. Use `web_search` to find candidate image pages.
2. Use `web_fetch` to inspect and extract direct image URLs.
3. Download with `exec` (`curl -L`) to local path, then send that path.
