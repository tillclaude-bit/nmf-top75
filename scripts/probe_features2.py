#!/usr/bin/env python3
"""Probe: gibt es kostenlose Spotify-Eigenschaften (Energy usw.) und Kuenstler-Genres von GitHub aus?"""
import csv, json, sys, urllib.request, urllib.error, os, base64

def get(url, headers=None):
    req = urllib.request.Request(url, headers=dict({"User-Agent": "nmf-top75/1.0", "Accept": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:
        return None, repr(e)

ids = [r["track_id"] for r in csv.DictReader(open("data/pools/pool_tracks_2026-11-06.csv", encoding="utf-8-sig"))][:30]
ids += ["7EtHVLkstCjz6iY79FmvLb"]  # Tove Styrke (Referenz: Spotify-Werte aus Apify bekannt)
print("ReccoBeats:")
s, b = get("https://api.reccobeats.com/v1/audio-features?ids=" + ",".join(ids))
print(" HTTP", s, b[:600].replace("\n", " "))
try:
    d = json.loads(b)
    items = d.get("content", d if isinstance(d, list) else [])
    print(" Treffer:", len(items), "von", len(ids))
    for it in items[:3]:
        print(" ", {k: it.get(k) for k in ("href", "energy", "valence", "danceability", "tempo", "loudness", "acousticness")})
except Exception as e:
    print(" parse", e)
print("Spotify Web API (Client Credentials):")
cid, sec = os.environ.get("SPOTIFY_CLIENT_ID", ""), os.environ.get("SPOTIFY_CLIENT_SECRET", "")
req = urllib.request.Request("https://accounts.spotify.com/api/token", data=b"grant_type=client_credentials",
        headers={"Authorization": "Basic " + base64.b64encode(f"{cid}:{sec}".encode()).decode(), "Content-Type": "application/x-www-form-urlencoded"})
tok = json.load(urllib.request.urlopen(req, timeout=30))["access_token"]
H = {"Authorization": "Bearer " + tok}
for path in (f"/tracks/{ids[0]}", f"/tracks?ids={','.join(ids[:5])}", f"/audio-features/{ids[0]}"):
    s, b = get("https://api.spotify.com/v1" + path, H)
    print(" ", path[:40], "HTTP", s, b[:160].replace("\n", " "))
s, b = get("https://api.spotify.com/v1" + f"/tracks/{ids[0]}", H)
try:
    aid = json.loads(b)["artists"][0]["id"]
    s, b = get(f"https://api.spotify.com/v1/artists/{aid}", H)
    print("  artist HTTP", s, b[:200].replace("\n", " "))
    s, b = get(f"https://api.spotify.com/v1/artists?ids={aid}", H)
    print("  artists-batch HTTP", s, b[:120].replace("\n", " "))
except Exception as e:
    print("  artist-probe", e)
