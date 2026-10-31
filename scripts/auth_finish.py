#!/usr/bin/env python3
"""Schritt 2: tauscht den Code aus der Weiterleitungs-URL gegen Tokens."""
import base64, json, os, sys, urllib.parse, urllib.request

cid = os.environ["SPOTIFY_CLIENT_ID"]
red = sys.argv[1]
q = urllib.parse.parse_qs(urllib.parse.urlparse(red).query)
st = json.load(open("auth/pkce.json"))
if q.get("error"):
    sys.exit(f"Spotify-Fehler: {q['error']}")
if q.get("state", [None])[0] != st["state"]:
    sys.exit("state passt nicht")
data = urllib.parse.urlencode({
    "grant_type": "authorization_code", "code": q["code"][0],
    "redirect_uri": "http://127.0.0.1:8888/callback",
    "client_id": cid, "code_verifier": st["verifier"]}).encode()
try:
    with urllib.request.urlopen(urllib.request.Request("https://accounts.spotify.com/api/token", data=data), timeout=30) as r:
        t = json.load(r)
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()[:300]}")
print("SCOPE=" + t.get("scope", ""))
print("REFRESH=" + t["refresh_token"][::-1])
