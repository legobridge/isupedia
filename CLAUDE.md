# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isupedia (aka "Ishanopedia II") is a Telegram bot that fetches Wikipedia summaries. Users send `/isu <topic>`, and the bot resolves the Wikipedia page (using OpenAI GPT for disambiguation when needed), returns a trimmed summary with a thumbnail photo and a shortened w.wiki URL.

## Architecture

Single-file Python application (`bot.py`):
- **Telegram layer**: `python-telegram-bot` async handlers for `/start` and `/isu` commands, wired up in `main()`
- **Wikipedia resolution**: `get_wiki_page()` attempts direct page fetch, falls back to search, and uses `disambiguate_using_gpt()` with concurrent summary fetching when disambiguation is needed
- **OpenAI integration**: Uses the Responses API (`client.responses.create`) with `gpt-5-mini` for disambiguation decisions
- **Thumbnail & URL**: Fetches page thumbnails via the MediaWiki API and shortens URLs via the Wikimedia `shortenurl` API

## Running

```bash
# Required environment variables
export TELEGRAM_BOT_TOKEN=...
export OPENAI_API_KEY=...

# Install dependencies
pip install -r requirements.txt

# Run the bot
python bot.py
```

## Deployment

Deployed on Fly.io (`fly.toml`, region: `iad`). Uses a Docker container built from `Dockerfile` (Python 3.12-slim). The bot runs as a long-polling process (not a webhook).

```bash
fly deploy
```
