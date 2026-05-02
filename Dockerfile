FROM python:3.12-slim-bookworm

WORKDIR /app

# Install deps as root, then drop privileges
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chmod=0644 bot.py .
COPY --chmod=0755 entrypoint.sh /entrypoint.sh

# Create a non-root user to run the bot
RUN useradd -u 1000 -M -s /bin/false botuser

USER botuser

ENTRYPOINT ["/entrypoint.sh"]
