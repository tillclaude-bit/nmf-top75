#!/bin/bash
# Verschluesselt den Refresh-Token mit dem Client Secret und legt ihn im (privaten) Repo ab.
set -e
mkdir -p auth
printf '%s' "$RT" | openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -a -pass env:SPOTIFY_CLIENT_SECRET > auth/refresh_token.enc
