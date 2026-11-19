# Islamic Stories Content Bot

Automated pipeline for creating verified Islamic storytelling content for TikTok. Sources content exclusively from the Quran and the six canonical Sunni hadith collections (Kutub al-Sittah), enforces all content rules programmatically, and gates everything through human review before publishing.

## Architecture

```
[Quran/Hadith/Seerah APIs] → [Grade/Denylist Filter] → [Narration Writer LLM]
    → [Citation Linter] → [TTS + Visual Generator] → [Video Assembler]
    → [Human Review Queue] → [TikTok Publisher]
```

Every stage has a hard gate. If sourcing fails, validation fails, the citation doesn't match, or review is rejected — the pipeline stops. It never fabricates content.

## Content Rules (Enforced in Code)

- **Source or nothing.** Every video is built from a verified Quran ayah or sahih/hasan hadith
- **Grade filter is programmatic.** Da'if (weak) and mawdu' (fabricated) are rejected by code, not by the LLM
- **Denylisted collections** — Al-Kafi, Bihar al-Anwar, Nahj al-Balagha are excluded at the data layer
- **Denylisted topics** — succession disputes, imamate, political parties, current events
- **No depiction of Prophets or Sahaba faces** — visuals are non-figurative (calligraphy, geometric, nature)
- **Every post carries an on-screen citation** and is disclosed as AI-generated

## Quick Start

### 1. Prerequisites

- Python 3.11+
- FFmpeg (for video assembly)
- API keys (see `.env.example`)

### 2. Install

```bash
# Clone and set up
cd islamic-stories-bot
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -e ".[dev]"
```

### 3. Configure

```bash
copy .env.example .env
# Edit .env with your API keys
```

### 4. Run

```bash
# Dry run (sourcing + narration + linting only, no media/publish)
python main_pipeline.py "People of the Cave" --source-type quran --dry-run

# Full run (generates video, enters review queue)
python main_pipeline.py "People of the Cave" --source-type quran

# Seerah topic
python main_pipeline.py "hijra to medina" --source-type seerah

# Review pending items
python main_pipeline.py --review

# Publish approved items
python main_pipeline.py --publish-approved
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
├── config/
│   ├── content_policy.yaml    # Content rules (collections, grades, denylists)
│   └── sources.yaml           # API endpoints and story mappings
├── src/
│   ├── sourcing/              # Quran, Hadith, Seerah data providers
│   ├── verification/          # Content validator + citation linter
│   ├── scripting/             # LLM narration writer
│   ├── media/                 # TTS, visual generator, video assembler
│   ├── review/                # Human review queue (SQLite + CLI)
│   └── publishing/            # TikTok API + browser fallback publishers
├── tests/                     # Unit tests for all critical paths
├── main_pipeline.py           # LangGraph orchestrator
└── .env.example               # API key template
```

## API Keys Required

| Service | Required? | How to Get |
|---------|-----------|------------|
| OpenAI or Anthropic | Yes | openai.com / anthropic.com |
| ElevenLabs | For TTS | elevenlabs.io |
| Sunnah.com | Optional | GitHub issue at github.com/sunnah-com/api |
| TikTok Developer | For publishing | developers.tiktok.com |

The hadith pipeline works without a sunnah.com key — it falls back to the fawazahmed0/hadith-api (free, no key needed). Quran.com's API requires no key for content endpoints.

## TikTok Compliance

- All posts set `is_ai_generated: true` via the Content Posting API
- Default privacy: `SELF_ONLY` until your app passes TikTok's audit
- Daily post limit: 2/day (configurable, well under TikTok's ~15/24h cap)
- Browser fallback **never** defeats CAPTCHA — logs and skips on bot detection

## License

MIT
"# tiktokpost_bot" 
