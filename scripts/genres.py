#!/usr/bin/env python3
"""Genre-Labels (und Deezer-Vorschau-URL) pro Song ueber die kostenlose Deezer-API.

Aufruf:
  python genres.py charts OUT.csv        # Chartsongs aus data/charts/song_features.csv
  python genres.py pool OUT.csv POOL.csv # pool_tracks_*.csv (track_id,name,artists)

Deezer-Genres haengen am Album (grob: Pop, Rock, Dance, Alternative, Rap/Hip Hop ...).
Alle Songs werden gleich behandelt; keine Zugangsdaten noetig.
"""
import csv
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (X11; Linux x86_64) nmf-top75/1.0"
ALBUM_CACHE = {}


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            if isinstance(data, dict) and data.get("error", {}).get("code") in (4, 100):
                time.sleep(2 + 2 * i)  # Quota
                continue
            return data
        except Exception:
            time.sleep(1 + i)
    return None


def norm(s):
    s = re.sub(r"[\(\[].*?[\)\]]", " ", s or "")
    s = re.split(r"\s-\s", s)[0]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def first_artist(a):
    return re.split(r",|&| feat\.| ft\.| x | with ", a or "")[0].strip()


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def lookup(row):
    tid, artist, title = row["id"], row["artist"], row["title"]
    out = {"id": tid, "artist": artist, "title": title, "dz_id": "", "dz_artist": "",
           "dz_title": "", "match": 0.0, "genres": "", "album_release": "", "dz_rank": "",
           "dz_preview": ""}
    fa, ct = first_artist(artist), re.sub(r"[\(\[].*?[\)\]]", "", title).strip()
    cands = []
    for q in (f'artist:"{fa}" track:"{ct}"', f"{fa} {ct}"):
        data = get("https://api.deezer.com/search?limit=8&q=" + urllib.parse.quote(q))
        cands = (data or {}).get("data") or []
        if cands:
            break
        time.sleep(0.15)
    best, best_s = None, 0.0
    for c in cands:
        s = 0.6 * ratio(norm(title), norm(c.get("title", ""))) + 0.4 * max(
            ratio(norm(fa), norm(c.get("artist", {}).get("name", ""))), 0.0
        )
        if s > best_s:
            best, best_s = c, s
    if not best or best_s < 0.8:
        out["match"] = round(best_s, 2)
        return out
    out.update({
        "dz_id": best["id"], "dz_artist": best["artist"]["name"], "dz_title": best["title"],
        "match": round(best_s, 2), "dz_rank": best.get("rank", ""),
        "dz_preview": best.get("preview", ""),
    })
    aid = best.get("album", {}).get("id")
    if aid:
        if aid not in ALBUM_CACHE:
            alb = get(f"https://api.deezer.com/album/{aid}") or {}
            ALBUM_CACHE[aid] = (
                ";".join(g["name"] for g in (alb.get("genres", {}) or {}).get("data", [])),
                alb.get("release_date", ""),
            )
        out["genres"], out["album_release"] = ALBUM_CACHE[aid]
    time.sleep(0.15)
    return out


def main():
    mode, out_path = sys.argv[1], sys.argv[2]
    rows = []
    if mode == "charts":
        with open("data/charts/song_features.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["match_quality"] in ("ok", "keine Audio-Features") and r["spotify_track_id"]:
                    rows.append({"id": r["spotify_track_id"],
                                 "artist": r["spotify_artist"] or r["chart_artist"],
                                 "title": r["spotify_name"] or r["chart_title"]})
    else:
        with open(sys.argv[3], encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({"id": r["track_id"], "artist": r["artists"], "title": r["name"]})
    print(f"{len(rows)} Songs", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lookup, rows))
    cols = list(res[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(res)
    matched = sum(1 for r in res if r["dz_id"])
    withgen = sum(1 for r in res if r["genres"])
    print(f"Deezer-Treffer: {matched} von {len(res)}, mit Genre: {withgen}")


if __name__ == "__main__":
    main()
