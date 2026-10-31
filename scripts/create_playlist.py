#!/usr/bin/env python3
"""Legt die Wochen-Playlist in Tills Spotify-Konto an (oeffentlich) und fuellt sie per Track-ID.

Aufruf: python create_playlist.py DATUM
Liest data/rankings/playlist_DATUM.csv. Token: auth/refresh_token.enc (mit SPOTIFY_CLIENT_SECRET verschluesselt).
"""
import csv, json, os, subprocess, sys, urllib.error, urllib.parse, urllib.request

date = sys.argv[1]
cid = os.environ["SPOTIFY_CLIENT_ID"]
rt = subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-a",
                     "-pass", "env:SPOTIFY_CLIENT_SECRET", "-in", "auth/refresh_token.enc"],
                    capture_output=True, text=True, check=True).stdout.strip()


def call(url, data=None, method=None, token=None, form=False):
    h = {}
    if token:
        h["Authorization"] = "Bearer " + token
    body = None
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode()
            h["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data).encode()
            h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


s, t = call("https://accounts.spotify.com/api/token",
            {"grant_type": "refresh_token", "refresh_token": rt, "client_id": cid}, form=True)
if s != 200:
    sys.exit(f"Token-Refresh fehlgeschlagen: {s} {t}")
tok = t["access_token"]
s, me = call("https://api.spotify.com/v1/me", token=tok)
print("me:", s, me.get("id") if isinstance(me, dict) else me)
ids = [r["track_id"] for r in csv.DictReader(open(f"data/rankings/playlist_{date}.csv", encoding="utf-8-sig"))]
name = f"Top 75 – {date}"
desc = "Automatisch aus den New-Music-Friday-Listen von 14 Ländern ausgewählt."
for url in ("https://api.spotify.com/v1/me/playlists", f"https://api.spotify.com/v1/users/{me.get('id')}/playlists"):
    s, pl = call(url, {"name": name, "public": True, "description": desc}, token=tok)
    print("create:", url.split("v1")[1], s, pl if s >= 300 else "ok")
    if s < 300:
        break
else:
    sys.exit("Playlist konnte nicht angelegt werden")
pid = pl["id"]
uris = ["spotify:track:" + i for i in ids]
for i in range(0, len(uris), 50):
    for ep in ("items", "tracks"):
        s, r = call(f"https://api.spotify.com/v1/playlists/{pid}/{ep}", {"uris": uris[i:i + 50]}, token=tok)
        print("add", i, ep, s, "" if s < 300 else r)
        if s < 300:
            break
s, chk = call(f"https://api.spotify.com/v1/playlists/{pid}", token=tok)
if s == 200:
    tot = (chk.get("tracks") or chk.get("items") or {}).get("total")
    print(f"Playlist {pid}: public={chk.get('public')} tracks={tot}")
print("URL=https://open.spotify.com/playlist/" + pid)
