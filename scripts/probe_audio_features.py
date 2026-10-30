#!/usr/bin/env python3
"""Einmaliger Test: Woher können wir Audio-Features pro Spotify-Track-ID bekommen?

Prüft nur Statuscodes und Feldnamen, gibt keine Zugangsdaten aus.
Erwartet die Umgebungsvariablen SPOTIFY_CLIENT_ID und SPOTIFY_CLIENT_SECRET.
"""
import base64
import json
import os
import urllib.error
import urllib.request

# alter Song (Kontrolle) + zwei neue Pool-Songs vom 30.10.2026
TRACKS = {
    "003vvx7Niy0yvhvHt4a68B": "The Killers - Mr. Brightside (alt, Kontrolle)",
    "2zUjgk6YpHn78MCzkj4R5B": "Gracie Abrams - Here You Come, There You Go (neu)",
    "3El0PldZjIxF1olHHzhUxW": "Dominic Fike - CASTING CALL (neu)",
}


def call(url, headers=None, data=None, method=None):
    req = urllib.request.Request(url, headers=headers or {}, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # Netzwerkfehler
        return None, repr(e)[:300]


def spotify_token():
    cid = os.environ["SPOTIFY_CLIENT_ID"]
    secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    status, body = call(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=b"grant_type=client_credentials",
        method="POST",
    )
    print(f"[Spotify] Token: HTTP {status}")
    if status != 200:
        print("  ", body[:200])
        return None
    return json.loads(body)["access_token"]


def main():
    token = spotify_token()
    if token:
        h = {"Authorization": f"Bearer {token}"}
        for tid, label in TRACKS.items():
            s, _ = call(f"https://api.spotify.com/v1/tracks/{tid}", headers=h)
            print(f"[Spotify] /tracks (Kontrolle Zugang) {label}: HTTP {s}")
        for tid, label in TRACKS.items():
            s, body = call(f"https://api.spotify.com/v1/audio-features/{tid}", headers=h)
            extra = ""
            if s == 200:
                extra = " Felder: " + ",".join(sorted(json.loads(body).keys()))
            print(f"[Spotify] /audio-features {label}: HTTP {s}{extra}")

    ids = ",".join(TRACKS)
    s, body = call(
        f"https://api.reccobeats.com/v1/audio-features?ids={ids}",
        headers={"Accept": "application/json"},
    )
    print(f"[ReccoBeats] audio-features fuer {len(TRACKS)} IDs: HTTP {s}")
    if s == 200:
        data = json.loads(body)
        items = data.get("content", data) if isinstance(data, dict) else data
        print(f"  Treffer: {len(items)} von {len(TRACKS)}")
        for it in items:
            keys = ",".join(sorted(k for k in it if k not in ("id", "href")))
            print("  ", it.get("href", it.get("id")), "->", keys)
            print(
                "   Werte:",
                {k: it.get(k) for k in ("tempo", "energy", "danceability", "valence", "acousticness", "loudness")},
            )
    else:
        print("  ", body[:200])


if __name__ == "__main__":
    main()
