# Pinned by digest so builds are reproducible and immune to tag-replacement.
# Update by re-pulling the tag and copying the new digest in:
#   docker pull python:3.12-slim-trixie
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim-trixie
#
# Using slim-trixie (Debian 13) over slim-bookworm (Debian 12) — newer
# baseline, fewer accumulated CVEs in distro packages.
#
# Fallback: if trixie ever introduces a Python wheel-compatibility issue or
# the CVE situation reverses, the previously-tested bookworm digest is:
#   python:3.12-slim-bookworm@sha256:d384a24aebd583543abb00fa47b2a434ec3a96559c6f244cc3cfcbe9e136a798
FROM python:3.12-slim-trixie@sha256:4386a385d81dba9f72ed72a6fe4237755d7f5440c84b417650f38336bbc43117

WORKDIR /app

# Install deps as root, then drop privileges
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all top-level Python modules. The bot is split across several files —
# bot.py is the entry point, the others are imported submodules.
COPY --chmod=0644 *.py ./
COPY --chmod=0755 entrypoint.sh /entrypoint.sh

# Create a non-root user to run the bot
RUN useradd -u 1000 -M -s /bin/false botuser

USER botuser

# Healthy ⇔ /tmp/signal-bot-healthy was touched within the last 90 seconds.
# `find -mmin -1.5` matches files modified in the last 90s. `[ -n ... ]` is
# true when find found one, false otherwise. signal-cli's WS pushes pings
# regularly, so a healthy bot refreshes the marker frequently.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD [ -n "$(find /tmp/signal-bot-healthy -mmin -1.5 2>/dev/null)" ] || exit 1

ENTRYPOINT ["/entrypoint.sh"]
