# Signal Ollama Bot

[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ed.svg)](https://docs.docker.com/compose/)
[![Local LLM](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com)

A Signal bot that maintains shared documents and answers questions, powered entirely by a local [Ollama](https://ollama.com) model. No API keys, no cloud services, no usage fees.

Each chat (group or DM) has its own independent document set, and admins can keep multiple named docs side-by-side per chat. Works for everyone in a claimed group, and as a private assistant in DMs for the owner.

## What it does

| Trigger | Where | Who | Action |
|---|---|---|---|
| `/doc` | Group + DM | Everyone (group) / owner (DM) | View default document |
| `/doc list` | Group + DM | Everyone | List all documents in this chat |
| `/doc status` | Group + DM | Everyone | Word count, line count, last modified |
| `/doc share` | Group + DM | Admins (group) / owner (DM) | Send default doc as a file |
| `/doc edit <instruction>` | Group + DM | Admins (group) / owner (DM) | AI-rewrites default doc |
| `/doc edit --long <instruction>` | Group + DM | Admins | Same, with a larger output cap (`LLM_LARGE_MAX_TOKENS`) |
| `/doc backup` | Group + DM | Admins (group) / owner (DM) | Manually snapshot current state to `.bak` (overwrites previous backup) |
| `/doc restore` | Group + DM | Admins (group) / owner (DM) | Undo the last edit by swapping with `.bak`; running it twice toggles back |
| `/doc create <name>` | Group + DM | Admins | Create a new named document |
| `/doc <name>` | Group + DM | Everyone | View named doc |
| `/doc <name> <verb>` *(view/status/share/edit/backup/restore/delete)* | Group + DM | Per-verb (admin for write/share/backup/restore) | Operate on named doc |
| `/group claim` | Group only | Owner | Approves the current group for bot use |
| `/group leave` | Group only | Owner | Removes the current group from approved set |
| `/admins` | Group + DM | Admins (group) / owner (DM) | List the owner and this group's admins |
| `/admins update` | Group only | Owner | Refresh the group's admin cache from Signal. Signal-side admins of *this* group become bot-admins for commands run *in this group only* — no automatic refresh, no spillover into other groups or DMs |
| `@bot <question>` *(native Signal mention)* | Group only | Everyone | Answers any question |
| *(any text)* | DM only | Owner | Treated as a question — no mention needed |

In DMs from the owner, every message is a query — no `@` prefix needed. In a group, the bot responds when it's @mentioned via Signal's native contact picker; the bot is identified by its UUID (`BOT_UUID`), independent of any nickname users have set.

---

## Requirements

- **Docker & Docker Compose** — [docker.com](https://docker.com)
- **Ollama** running locally — [ollama.com](https://ollama.com)
- **A phone number for the bot** — VoIP numbers (Twilio, Google Voice) work fine

No API keys required.

---

## Setup

### 1. Pull an Ollama model

```bash
ollama pull llama3.3
ollama run llama3.3 "say hello"   # smoke test
```

| RAM | Recommended model |
|---|---|
| 8 GB  | `llama3.2` or `qwen2.5:7b` |
| 16 GB | `llama3.3` or `qwen2.5:14b` |
| 32 GB+ | `qwen2.5:32b` or `llama3.3:70b` |

> **Mac:** Install Ollama natively (not in Docker) so it can use Metal GPU acceleration. The bot reaches it via `host.docker.internal`.

### 2. Configure

```bash
git clone <this-repo> && cd signal-bot
cp env.example .env
mkdir -p secrets && echo "+15550000000" > secrets/bot_number.txt && chmod 600 secrets/bot_number.txt
```

Replace `+15550000000` with the bot's phone number. The secret file is mounted into the container at runtime — `BOT_NUMBER` is not stored in `.env`.

### 3. Register the bot's phone number with Signal

```bash
docker compose up -d signal-api
```

Then either **register a fresh number** (recommended for a dedicated bot) or **link to an existing account** as a secondary device.

**Register a fresh number:**

Signal requires a captcha for new registrations. Visit <https://signalcaptchas.org/registration/generate.html>, solve the captcha, then right-click the resulting **"Open Signal"** link and copy the URL — it looks like `signalcaptcha://signal-hcaptcha.<TOKEN>`.

```bash
# Submit the captcha to request an SMS code
curl -X POST http://localhost:8080/v1/register/+15550000000 \
  -H 'Content-Type: application/json' \
  -d '{"captcha":"signalcaptcha://signal-hcaptcha.YOURTOKEN"}'

# Submit the SMS code you received (format: 123-456)
curl -X POST http://localhost:8080/v1/register/+15550000000/verify/123-456 \
  -H 'Content-Type: application/json'
```

> Captcha tokens expire quickly — within a few minutes — so generate it just before running the first command.

**Or link to an existing Signal account** as a secondary device (no captcha needed):

```bash
open http://localhost:8080/v1/qrcodelink?device_name=signal-bot
# Scan the QR with: Signal app → Settings → Linked Devices → +
```

> The `signal-api` port mapping in `docker-compose.yml` is bound to `127.0.0.1` only, so the API is never reachable from outside the host. You can leave it as-is.

### 4. Set a profile name on the bot

Without this, Signal can silently drop incoming messages from new contacts.

```bash
curl -X PUT -H 'Content-Type: application/json' \
  -d '{"name":"Bot"}' \
  http://localhost:8080/v1/profiles/+15550000000
```

### 5. Start the bot and discover your IDs

```bash
docker compose up -d
docker compose logs -f bot
```

You'll see a startup banner. The bot is now listening but doesn't know about any groups yet.

**Find your owner ID:** DM the bot anything (or send any message in a group with the bot present). The bot will log a line like:

```
DM source=<your-id>
```

Copy `<your-id>` into `OWNER_ID` in `.env`. That's either your phone number or your UUID, whichever signal-cli has for you.

**Adding more admins** is a per-group action — make them admins in Signal's own group settings, then run `/admins update` in that group. Their bot-admin status is scoped to that group only. (See [Configuration knobs](#configuration-knobs) and the [admins commands](#what-it-does) above.)

**Find the bot's own UUID** *(required for group @mentions)*: in the group, type `@` and select the bot from Signal's contact picker, then send. The bot logs:

```
Mentions in message: ['<bot-uuid>']
```

Copy that UUID into `BOT_UUID` in `.env` and force-recreate the container. Once set, the bot responds to native @mentions of itself in any approved group. Without `BOT_UUID`, the bot won't respond to mentions at all — only `/doc` commands work.

**Add to your group with explicit approval.** Once `OWNER_ID` is set:

1. Add the bot to your Signal group.
2. Send `/group claim` in the group (only the owner's account is allowed to run this).
3. The bot replies `✅ Group claimed.` and persists the group to `data/groups.txt`.

From that point on, the bot responds to `/doc`, `@mention`, etc. in that group. Repeat for any group you want the bot to participate in.

To revoke a group, send `/group leave` in the group as the owner — the bot replies `👋 Group released.` and stops responding there.

> **Why explicit instead of automatic?** Anyone in your contacts can add you to a Signal group. If group approval were silent, an attacker could invite you and the bot to a group and then prompt the bot at will. Requiring you to explicitly run `/group claim` keeps that surface closed.

### 6. Activate

After editing `.env`, recreate the bot container so it picks up the new values:

```bash
docker compose up -d --force-recreate bot
docker compose logs -f bot
```

Try one of:

- **Group**: type `@` in your group chat, pick the bot from Signal's autocomplete, then send your question.
- **DM**: if you set `OWNER_ID`, just DM the bot any question — no prefix needed.

Reply should arrive within a few seconds.

---

## Customisation

**Bot's display name** — set the Signal profile name on the bot's account (step 4 in setup). That's the name users see when @-mentioning. Internal matching uses `BOT_UUID`, so you can change the display name freely without touching `.env`.

**Different models for edit vs query**:
```
MODEL_EDIT=qwen2.5:14b      # stronger instruction-following for edits
MODEL_SEARCH=llama3.2       # smaller/faster model for Q&A
```

**Web search behaviour** — `SEARCH_FALLBACK` in `.env`:

| Value | Behaviour |
|---|---|
| *(unset / empty — what `env.example` ships with)* | Training knowledge only, with a stale-data warning footer. Fully local — no outbound HTTP. |
| `auto` | Keyword heuristic — searches DuckDuckGo only on time-sensitive queries (price, news, latest, current, etc.) |
| `duckduckgo` | Always searches before answering |

The keyword list is at the top of [bot.py](bot.py) under `_TIMELY_PATTERN` — edit it freely.

> **Search is opt-in.** The `ddgs` Python package is **commented out** in [requirements.txt](requirements.txt) by default to keep the dependency surface small. If you set `SEARCH_FALLBACK=auto` or `duckduckgo`, uncomment the `ddgs` line and rebuild — otherwise search calls fail silently and the bot answers from training knowledge.

**Customise `/doc edit` prompt** — `EDIT_SYSTEM_PROMPT` in [bot.py](bot.py).

**Bot persona and personal context** — edit `data/persona.md` (auto-created on first run with a starter template). It's used as the system prompt for every Q&A reply, so this is where you tell the bot:

- **Who it is** — give it a name and stop it from outing itself as "an LLM trained by Google" or whatever the underlying model defaults to.
- **Who you are** — when you say "my brother" or "the cabin" or "Mom", the bot will know what you mean.
- **Group context** — names of group members, ongoing topics, anything else worth keeping in mind.

Changes are picked up on the very next message — no restart needed.

**Multiple documents per chat** — each group chat (and the DM context) has its own independent document namespace under `data/docs/`. Within a chat, admins can manage as many named docs as they want:

```
/doc list                              # show all docs in this chat
/doc create recipes                    # create a new named doc
/doc recipes edit add a carbonara recipe
/doc recipes share                     # send recipes.md as a file
/doc recipes delete                    # remove the doc (admin)
```

Doc names use letters, digits, `_`, `-` (1–40 chars). The default doc is auto-created the first time you run `/doc edit` in a chat.

**Long edits** — by default each `/doc edit` is capped at `LLM_MAX_TOKENS` (≈12K chars output) to bound runaway prompts. Admins can opt into a larger cap per call:

```
/doc edit --long restructure into 6 detailed sections
```

`--long` raises the cap to `LLM_LARGE_MAX_TOKENS` (≈40K chars by default) for that one edit only.

**Plain-prose replies** — the bot strips disruptive markdown (headers, bold, italic asterisks, code fences, link syntax) from Q&A responses so they read cleanly in Signal. Document content is left as markdown.

---

## Configuration knobs

All optional, all in `.env`. Defaults in [env.example](env.example).

| Variable | Default | Purpose |
|---|---|---|
| `MAX_INLINE_CHARS` | `1500` | `/doc` view truncation point — beyond this, send via `/doc share` |
| `MAX_INSTRUCTION` | `500` | Cap on the user's instruction text for `/doc edit` |
| `MAX_QUERY_CHARS` | `1000` | Cap on `@bot` Q&A input length |
| `MAX_DOC_CHARS` | `50000` | Hard ceiling on persisted doc size |
| `MAX_DOC_INPUT_CHARS` | `20000` | Largest doc accepted as input to `/doc edit` |
| `LLM_MAX_TOKENS` | `3000` | Per-call output cap (~12K chars) |
| `LLM_LARGE_MAX_TOKENS` | `10000` | Output cap for admin `--long` edits (~40K chars) |
| `LLM_TIMEOUT` | `180` | Wall-clock timeout per Ollama call (seconds) |
| `PROGRESS_NUDGE_SECONDS` | `30` | Seconds before sending "still working" message; set `0` to disable |
| `RATE_LIMIT_MAX` | `10` | Max messages per `RATE_LIMIT_WINDOW` per sender (owner exempt); `0` to disable |
| `RATE_LIMIT_WINDOW` | `60` | Sliding window in seconds for the rate limit |
| `SEARCH_FALLBACK` | *empty* | Web search mode: empty (off), `auto`, or `duckduckgo` |

---

## Security posture

What's hardened:

- **Container runs as unprivileged user**, with a read-only root filesystem (`/tmp` is the only writable mount), all Linux capabilities dropped, `no-new-privileges`, and CPU/RAM limits.
- **Per-sender rate limiting** (`RATE_LIMIT_MAX`/`RATE_LIMIT_WINDOW`) on every inbound message, owner exempt.
- **Watchdog timeout** (`LLM_TIMEOUT`) on every Ollama call, enforced from the bot's own thread so a stalled model can't hang the bot.
- **Output caps** on every LLM call (`LLM_MAX_TOKENS`) bound runaway generations; an explicit admin `--long` flag is the only way past the default.
- **Pre-edit document backup** — `/doc edit` snapshots the current doc to `<name>.md.bak` before overwriting. Bad LLM edits are recoverable by `cp <name>.md.bak <name>.md`.
- **Edit-summary line delta** in every `/doc edit` reply makes silent rewrites hard to miss.
- **Prompt-injection sanitiser** strips role markers (`system:`, `user:`, `assistant:`) from every user-supplied LLM input.
- **Markdown stripping** on chat replies prevents the model from leaking literal formatting noise.
- **Group approval is explicit** — `/group claim` from the owner is required before the bot responds in any group; no implicit auto-onboarding.
- **Search is off by default** — fully local. The `ddgs` package isn't even installed unless you uncomment it.

Third-party image CVEs from `signal-cli-rest-api` are tracked separately. See [CVE-ASSESSMENT.md](CVE-ASSESSMENT.md) for the latest scan results, why none of the findings affect this deployment, and the re-scan schedule.

What's intentionally not hardened:

- Documents are stored unencrypted on disk. If the host filesystem is compromised, so is your data.
- `/doc` (view) is open to all group members of an approved group. Don't put data in there that isn't appropriate for everyone in the group to see.
- Admins are trusted with whatever they can write into a doc via `/doc edit`. The trust boundary is per-group: a Signal-recognized admin in Group A only has bot-admin power in Group A. Demoting them on Signal's side and running `/admins update` in that group revokes it.

---

## Troubleshooting

**`receive() failed` or no DMs arriving** — usually a profile/contact issue. Confirm step 4 ran successfully. If the bot's account is brand-new, send it a message from yourself first to establish a session.

**`/doc edit` produced something terrible** — every edit snapshots the previous version to `<name>.md.bak` in the same directory. Recover with `cp ./data/docs/<context>/<name>.md.bak ./data/docs/<context>/<name>.md`. The backup is overwritten on the next edit, so save copies somewhere else if you might need an older version.

**`send() HTTP 400 ... "Invalid identifier"`** — the bot's group recipient construction is wrong. The wrapper expects `group.<base64(internal_id)>`; this is handled automatically, but if you've modified the routing code, check [bot.py](bot.py)'s `route()` function.

**Bot replies in group don't reach you, but DMs work**, or you can't be added to the group via your phone number — your Signal account has **"Find by Phone Number"** disabled. The bot must address you by username or UUID instead. Use your UUID (visible in any envelope's `sourceUuid` or `source`) for `OWNER_ID`. Bot-to-non-findable-user sends require the username, e.g. `recipients=["yourname.42"]`.

**`@bot` in the group produces no log line** — make sure `BOT_UUID` is set in `.env` (see setup step 5). Without it, the bot ignores all mentions. Also confirm you're using Signal's native @mention picker (typing `@` and selecting from the list) rather than typing the name as plain text.

**Edited `.env` but the bot doesn't see the new value** — `docker compose restart` reuses the existing container with old env. Use `docker compose up -d --force-recreate bot` to actually recreate it. Verify with `docker compose exec bot env | grep VAR_NAME`.

---

## File structure

```
signal-bot/
├── bot.py              # main bot logic (websocket consumer + handlers)
├── docker-compose.yml  # signal-api + bot + secrets wiring
├── Dockerfile          # builds the bot container
├── entrypoint.sh       # loads BOT_NUMBER from the Docker secret
├── requirements.txt
├── env.example         # copy to .env and fill in
├── secrets/
│   └── bot_number.txt  # bot's phone number — Docker secret
├── data/
│   ├── persona.md          # bot identity + relational context (auto-created)
│   ├── groups.txt          # claimed group IDs (managed via /group claim)
│   ├── group_admins.json   # per-group Signal admin cache (managed via /admins update)
│   └── docs/               # per-chat document namespaces
│       ├── dm/             # owner-DM documents
│       │   ├── default.md
│       │   └── default.md.bak    # pre-edit backup (overwritten each edit)
│       └── <group-key>/    # one subdir per claimed group
│           ├── default.md
│           ├── meeting-notes.md
│           └── meeting-notes.md.bak
└── signal-config/      # Signal keys, owned by signal-api (auto-created)
```

---

## Hosting options

| Option | Notes |
|---|---|
| **Mac (local)** | Ollama runs natively with Metal GPU. Run Docker Desktop alongside. |
| **Linux server / VPS** | Best for GPU-accelerated Ollama. $5–10/mo for a basic VPS, BYO GPU for serious use. |
| **Raspberry Pi** | Pi 4/5 with smaller models (`llama3.2`, `qwen2.5:7b`). CPU-only, slower replies. |
