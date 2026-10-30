#!/usr/bin/env python3
"""Spotify-Kuenstlerprofile (Genres, Follower, Popularitaet) fuer Chart- und Pool-Songs.

Aufruf: python artist_genres.py DATUM OUT.csv
Nutzt SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET (Client-Credentials, keine Nutzerrechte).
Gleiche Abfrage fuer alle Songs, Hauptkuenstler = erster Kuenstler des Tracks.
"""
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.spotify.com/v1"
STATS = {"429": 0, "err": 0}


def token():
    cid, sec = os.environ["SPOTIFY_CLIENT_ID"], os.environ["SPOTIFY_CLIENT_SECRET"]
    auth = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=b"grant_type=client_credentials",
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["access_token"]


TOKEN = None


def get(path):
    global TOKEN
    for i in range(8):
        req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {TOKEN}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                STATS["429"] += 1
                time.sleep(min(int(e.headers.get("Retry-After", "2")) + 1, 30))
                continue
            if e.code == 401:
                TOKEN = token()
                continue
            return e.code, None
        except Exception:
            STATS["err"] += 1
            time.sleep(1 + i)
    return None, None


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main():
    global TOKEN
    date, out = sys.argv[1], sys.argv[2]
    TOKEN = token()
    ids = []
    with open("data/charts/song_features.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["match_quality"] in ("ok", "keine Audio-Features") and r["spotify_track_id"]:
                ids.append(r["spotify_track_id"])
    with open(f"data/pools/pool_tracks_{date}.csv", encoding="utf-8-sig") as f:
        ids += [r["track_id"] for r in csv.DictReader(f)]
    ids = list(dict.fromkeys(ids))
    print(f"{len(ids)} Tracks", flush=True)

    t2a = {}
    batch_ok = True
    for ch in chunks(ids, 50):
        done = False
        if batch_ok:
            s, d = get("/tracks?ids=" + ",".join(ch))
            if s == 200 and d:
                for t in d["tracks"]:
                    if t and t.get("artists"):
                        t2a[t["id"]] = (t["artists"][0]["id"], t["artists"][0]["name"])
                done = True
            else:
                batch_ok = False
                print(f"Batch /tracks nicht verfuegbar (HTTP {s}), nutze Einzelabrufe", flush=True)
        if not done:
            for tid in ch:
                s, d = get(f"/tracks/{tid}")
                if s == 200 and d and d.get("artists"):
                    t2a[tid] = (d["artists"][0]["id"], d["artists"][0]["name"])
                time.sleep(0.05)
    aids = list(dict.fromkeys(a for a, _ in t2a.values()))
    print(f"{len(t2a)} Tracks mit Kuenstler, {len(aids)} Kuenstler", flush=True)

    info = {}
    batch_ok = True
    for ch in chunks(aids, 50):
        done = False
        if batch_ok:
            s, d = get("/artists?ids=" + ",".join(ch))
            if s == 200 and d:
                for a in d["artists"]:
                    if a:
                        info[a["id"]] = a
                done = True
            else:
                batch_ok = False
                print(f"Batch /artists nicht verfuegbar (HTTP {s}), nutze Einzelabrufe", flush=True)
        if not done:
            for aid in ch:
                s, d = get(f"/artists/{aid}")
                if s == 200 and d:
                    info[aid] = d
                time.sleep(0.05)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "artist_id", "artist_name", "genres", "followers", "popularity"])
        for tid, (aid, name) in t2a.items():
            a = info.get(aid, {})
            w.writerow([tid, aid, name, ";".join(a.get("genres", [])),
                        (a.get("followers") or {}).get("total", ""), a.get("popularity", "")])
    with_g = sum(1 for tid, (aid, _) in t2a.items() if info.get(aid, {}).get("genres"))
    print(f"Kuenstlerprofile: {len(info)} von {len(aids)}; Tracks mit Genre: {with_g} von {len(t2a)}")
    print(f"Statistik: {STATS}")


if __name__ == "__main__":
    main()
