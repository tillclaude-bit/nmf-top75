#!/usr/bin/env python3
"""Liest eine Playlist aus Tills Konto (Reihenfolge = sein Ranking) und schreibt sie als CSV.
Aufruf: python read_playlist.py PLAYLIST_ID OUT.csv"""
import csv, json, os, subprocess, sys, urllib.error, urllib.parse, urllib.request

pid, out = sys.argv[1], sys.argv[2]
cid = os.environ["SPOTIFY_CLIENT_ID"]
rt = subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-a",
                     "-pass", "env:SPOTIFY_CLIENT_SECRET", "-in", "auth/refresh_token.enc"],
                    capture_output=True, text=True, check=True).stdout.strip()


def call(url, data=None, token=None):
    h = {"Authorization": "Bearer " + token} if token else {}
    body = urllib.parse.urlencode(data).encode() if data else None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


s, t = call("https://accounts.spotify.com/api/token", {"grant_type": "refresh_token", "refresh_token": rt, "client_id": cid})
if s != 200:
    sys.exit(f"Token: {s} {t}")
tok = t["access_token"]
rows, url = [], f"https://api.spotify.com/v1/playlists/{pid}/items?limit=50"
while url:
    s, d = call(url, token=tok)
    if s != 200:
        s2, d = call(url.replace("/items", "/tracks"), token=tok)
        if s2 != 200:
            sys.exit(f"Lesen fehlgeschlagen: {s} {d}")
    for it in d["items"]:
        tr = it.get("item") or it.get("track") or {}
        if tr.get("id"):
            rows.append({"pos": len(rows) + 1, "track_id": tr["id"], "name": tr["name"],
                         "artists": ";".join(a["name"] for a in tr["artists"]), "added_at": it.get("added_at", "")})
    url = d.get("next")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["pos", "track_id", "name", "artists", "added_at"])
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} Songs gelesen")
