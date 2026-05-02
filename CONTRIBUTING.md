# Contributing

This is a small, single-file Signal bot project. Contributions are welcome — keep them focused.

## Before you start

For anything beyond a typo or one-line fix, **open an issue first**. A 2-line description of what you're trying to change saves both of us time if the answer is "this is intentional" or "we'd rather solve it differently."

## Local development

The bot is a single Python file plus a docker-compose stack. To iterate:

```bash
# Edit bot.py on the host
docker compose build bot && docker compose up -d --force-recreate bot
docker compose logs -f bot
```

Volume-mounted files in `./data/` (`document.md`, `persona.md`, `groups.txt`) survive rebuilds, so you don't lose state between iterations.

## Quick checks before submitting

```bash
# Syntax check
python3 -m py_compile bot.py

# Compose file validates and renders
docker compose config -q
```

## What we'd like to see in PRs

- **One concern per PR.** A bug fix and an unrelated refactor should be two PRs.
- **Match existing style.** No type-hint overhauls, no new dependencies, no abstractions for hypothetical future use cases. The bot is deliberately simple.
- **Update the README** if you change user-facing behavior (commands, env vars, setup steps).
- **Update `env.example`** if you add or rename an env var.
- **Don't commit `secrets/`, `.env`, `data/`, or `signal-config/`.** The `.gitignore` should catch these — verify with `git status` before pushing.

## What's likely to be declined

- Pulling in heavy dependencies (LLM frameworks, ORMs, queue libraries). The point is "small Python script + Ollama."
- Cloud-only features. The local-first posture is intentional.
- Speculative refactors that don't fix a real bug or add a real feature.
