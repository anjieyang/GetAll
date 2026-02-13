<div align="center">
  <img src="getall_logo.png" alt="GetAll" width="500">
  <h1>GetAll: AI-Powered Bitget Trading Assistant</h1>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

**GetAll** is an AI-native crypto trading assistant built on top of the [nanobot](https://github.com/HKUDS/nanobot) framework, deeply integrated with the **Bitget** exchange.

- Real-time market data, portfolio tracking, and trade execution via Bitget API
- Natural language trading commands ("帮我开多 BTC 100U 20x")
- Strategy builder, backtesting, and risk management
- Telegram bot with inline trading UI
- Multi-provider LLM support (OpenRouter, Claude, GPT, DeepSeek, etc.)

## 🏗️ Architecture

<p align="center">
  <img src="getall_arch.png" alt="GetAll architecture" width="800">
</p>

## 📦 Install

```bash
git clone https://github.com/anjieyang/GetAll.git
cd GetAll
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## 🚀 Quick Start

**1. Initialize**

```bash
getall onboard
```

**2. Configure** (`~/.getall/config.json`)

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

**3. Chat**

```bash
getall agent -m "BTC 现在什么价？"
```

## 💬 Telegram Bot

**1. Create a bot** — search `@BotFather` in Telegram, send `/newbot`, copy the token.

**2. Configure**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

**3. Run**

```bash
getall gateway
```

<details>
<summary><b>Other Channels (Discord, WhatsApp, Feishu, Slack, Email, QQ, DingTalk)</b></summary>

| Channel | Setup |
|---------|-------|
| **Discord** | Easy (bot token + intents) |
| **WhatsApp** | Medium (scan QR) |
| **Feishu** | Medium (app credentials) |
| **DingTalk** | Medium (app credentials) |
| **Slack** | Medium (bot + app tokens) |
| **Email** | Medium (IMAP/SMTP credentials) |
| **QQ** | Easy (app credentials) |

Refer to channel-specific setup in `getall/channels/` source code or run `getall gateway` with the corresponding config block.

</details>

## ⚙️ Configuration

Config file: `~/.getall/config.json`

### Providers

| Provider | Purpose | Get API Key |
|----------|---------|-------------|
| `openrouter` | LLM (recommended, access to all models) | [openrouter.ai](https://openrouter.ai) |
| `anthropic` | LLM (Claude direct) | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | LLM (GPT direct) | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | LLM (DeepSeek direct) | [platform.deepseek.com](https://platform.deepseek.com) |
| `groq` | LLM + Voice transcription (Whisper) | [console.groq.com](https://console.groq.com) |
| `gemini` | LLM (Gemini direct) | [aistudio.google.com](https://aistudio.google.com) |
| `vllm` | LLM (local, any OpenAI-compatible server) | — |

### Bitget API

Set the following environment variables (or in `.env`):

```
BITGET_API_KEY=xxx
BITGET_API_SECRET=xxx
BITGET_API_PASSPHRASE=xxx
```

### Security

| Option | Default | Description |
|--------|---------|-------------|
| `tools.restrictToWorkspace` | `false` | Restrict agent tools to workspace directory |
| `channels.*.allowFrom` | `[]` (allow all) | Whitelist of user IDs |

## CLI Reference

| Command | Description |
|---------|-------------|
| `getall onboard` | Initialize config & workspace |
| `getall agent -m "..."` | Chat with the agent |
| `getall agent` | Interactive chat mode |
| `getall gateway` | Start the gateway (Telegram, etc.) |
| `getall status` | Show status |
| `getall channels login` | Link WhatsApp (scan QR) |
| `getall cron add` | Add a scheduled task |
| `getall cron list` | List scheduled tasks |

## 🐳 Docker

```bash
docker build -t getall .

# Initialize (first time)
docker run -v ~/.getall:/root/.getall --rm getall onboard

# Run gateway
docker run -v ~/.getall:/root/.getall -p 18790:18790 getall gateway

# Single command
docker run -v ~/.getall:/root/.getall --rm getall agent -m "Hello!"
```

## 📁 Project Structure

```
getall/
├── agent/          # 🧠 Core agent logic (loop, context, memory, tools)
├── trading/        # 📈 Bitget trading (market data, orders, backtest, strategies)
├── integrations/   # 🔗 Exchange gateway (Bitget API wrapper)
├── nl/             # 🗣️ Natural language intent engine
├── channels/       # 📱 Chat channels (Telegram, Discord, etc.)
├── skills/         # 🎯 Bundled skills (trading strategies, risk check, etc.)
├── providers/      # 🤖 LLM providers (OpenRouter, Claude, GPT, etc.)
├── memory/         # 💾 Canonical memory store & recall
├── storage/        # 🗄️ SQLite persistence layer
├── api/            # 🌐 REST API endpoints
├── bus/            # 🚌 Message routing
├── cron/           # ⏰ Scheduled tasks
├── session/        # 💬 Conversation sessions
├── config/         # ⚙️ Configuration
└── cli/            # 🖥️ CLI commands
```

## License

MIT
