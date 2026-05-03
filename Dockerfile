# Pinned by digest so builds are reproducible and immune to tag-replacement.
# Update by re-pulling the tag and copying the new digest in:
#   docker pull python:3.12-slim-bookworm
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim-bookworm
FROM python:3.12-slim-bookworm@sha256:d384a24aebd583543abb00fa47b2a434ec3a96559c6f244cc3cfcbe9e136a798

WORKDIR /app

# Install deps as root, then drop privileges
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chmod=0644 bot.py .
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
