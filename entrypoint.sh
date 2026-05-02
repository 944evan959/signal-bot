#!/bin/sh
# Load Docker secrets from files into environment variables.
# Docker mounts secrets at /run/secrets/<name>.

if [ -f /run/secrets/bot_number ]; then
    export BOT_NUMBER=$(cat /run/secrets/bot_number)
fi

exec python -u bot.py
