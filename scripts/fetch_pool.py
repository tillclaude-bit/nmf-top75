#!/usr/bin/env python3
"""Holt den New-Music-Friday-Pool von Apify und archiviert ihn komplett im Repo.

Aufruf:
  python fetch_pool.py DATUM                 # startet den Scraper neu (ca. 3 Cent)
  python fetch_pool.py DATUM --dataset ID    # nutzt ein vorhandenes Dataset (kostenlos)

Benoetigt APIFY_TOKEN als Umgebungsvariable (GitHub-Secret).
Schreibt:
  data/pools/pool_raw_DATUM.csv     alle Zeilen (Playlist x Track), unveraendert
  data/pools/pool_tracks_DATUM.csv  ein Eintrag pro Track mit Playlist-Breite und Positionen
"""
import argparse
import csv
import json
import os
import urllib.parse
import urllib.request

ACTOR = "automation-lab~spotify-playlist-search-scraper"
API = "https://api.apify.com/v2"


def req(url, data=None):
    r = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.load(resp)


def run_actor(token):
    with open("config/pool_scraper_input.json", encoding="utf-8") as f:
        payload = json.dumps(json.load(f)).encode()
    url = f"{API}/acts/{ACTOR}/run-sync-get-dataset-items?token={token}&format=json&clean=true"
    return req(url, payload)


def read_dataset(token, dataset_id):
    items, offset = [], 0
    while True:
        q = urllib.parse.urlencode(
            {"token": token, "format": "json", "clean": "true", "offset": offset, "limit": 1000}
        )
        page = req(f"{API}/datasets/{dataset_id}/items?{q}")
        items += page
        if len(page) < 1000:
            return items
        offset += 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--dataset")
    a = ap.parse_args()
    token = os.environ["APIFY_TOKEN"]
    items = read_dataset(token, a.dataset) if a.dataset else run_actor(token)
    items = [i for i in items if i.get("type", "track") == "track" and i.get("trackId")]
    print(f"{len(items)} Zeilen (Playlist x Track)")

    os.makedirs("data/pools", exist_ok=True)
    raw_cols = [
        "playlistId", "playlistName", "position", "trackId", "name", "artists",
        "durationMs", "explicit", "playable", "previewUrl",
    ]
    with open(f"data/pools/pool_raw_{a.date}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)

    tracks = {}
    for i in items:
        t = tracks.setdefault(
            i["trackId"],
            {"track_id": i["trackId"], "name": i["name"], "artists": i["artists"],
             "duration_ms": i.get("durationMs"), "explicit": i.get("explicit"),
             "preview_url": i.get("previewUrl") or "", "playlists": [], "positions": []},
        )
        t["playlists"].append(i["playlistName"])
        t["positions"].append(int(i["position"]))
        if not t["preview_url"] and i.get("previewUrl"):
            t["preview_url"] = i["previewUrl"]
    cols = ["track_id", "name", "artists", "duration_ms", "explicit", "n_playlists",
            "best_pos", "mean_pos", "playlists", "preview_url"]
    with open(f"data/pools/pool_tracks_{a.date}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in tracks.values():
            w.writerow({
                **{k: t[k] for k in ("track_id", "name", "artists", "duration_ms", "explicit", "preview_url")},
                "n_playlists": len(t["playlists"]),
                "best_pos": min(t["positions"]),
                "mean_pos": round(sum(t["positions"]) / len(t["positions"]), 1),
                "playlists": ";".join(t["playlists"]),
            })
    print(f"{len(tracks)} eindeutige Tracks, davon ohne Preview-URL: "
          f"{sum(1 for t in tracks.values() if not t['preview_url'])}")


if __name__ == "__main__":
    main()
