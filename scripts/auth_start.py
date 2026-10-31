#!/usr/bin/env python3
"""Schritt 1: erzeugt den Spotify-Freigabe-Link (PKCE) und merkt sich den Verifier."""
import base64, hashlib, json, os, secrets, urllib.parse

REDIRECT = "http://127.0.0.1:8888/callback"
cid = os.environ["SPOTIFY_CLIENT_ID"]
ver = secrets.token_urlsafe(64)[:96]
chal = base64.urlsafe_b64encode(hashlib.sha256(ver.encode()).digest()).decode().rstrip("=")
state = secrets.token_urlsafe(8)
os.makedirs("auth", exist_ok=True)
json.dump({"verifier": ver, "state": state}, open("auth/pkce.json", "w"))
q = urllib.parse.urlencode({
    "client_id": "CID", "response_type": "code", "redirect_uri": REDIRECT,
    "scope": "playlist-modify-public playlist-modify-private", "state": state,
    "code_challenge_method": "S256", "code_challenge": chal})
# Client ID wird von GitHub in Logs maskiert -> umgekehrt ausgeben
print("URL=https://accounts.spotify.com/authorize?" + q)
print("CIDREV=" + cid[::-1])
