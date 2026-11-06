#!/usr/bin/env python3
import csv, json, os, base64, urllib.request, urllib.error

def get(url, headers=None):
    req = urllib.request.Request(url, headers=dict({"User-Agent": "nmf-top75/1.0", "Accept": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:
        return None, repr(e)

out = []
ids = [r["track_id"] for r in csv.DictReader(open("data/pools/pool_tracks_2026-11-06.csv", encoding="utf-8-sig"))][:6]
ids += ["7EtHVLkstCjz6iY79FmvLb", "4cOdK2wGLETKBW3PvgPWqT"]  # Tove Styrke; bekannter Hit als Kontrolle
one = ids[-1]
tests = {
 "rb_single_track": "https://api.reccobeats.com/v1/track?ids=" + one,
 "rb_repeat_param": "https://api.reccobeats.com/v1/track?" + "&".join("ids=" + i for i in ids[-2:]),
 "rb_features_spotifyid": "https://api.reccobeats.com/v1/audio-features?ids=" + one,
 "rb_features_url": "https://api.reccobeats.com/v1/audio-features?ids=" + "https%3A%2F%2Fopen.spotify.com%2Ftrack%2F" + one,
}
for k, u in tests.items():
    s, b = get(u)
    out.append(f"{k}: HTTP {s} {b[:220]}".replace("\n", " "))
cid, sec = os.environ["SPOTIFY_CLIENT_ID"], os.environ["SPOTIFY_CLIENT_SECRET"]
req = urllib.request.Request("https://accounts.spotify.com/api/token", data=b"grant_type=client_credentials",
        headers={"Authorization": "Basic " + base64.b64encode(f"{cid}:{sec}".encode()).decode(), "Content-Type": "application/x-www-form-urlencoded"})
H = {"Authorization": "Bearer " + json.load(urllib.request.urlopen(req, timeout=30))["access_token"]}
s, b = get("https://api.spotify.com/v1/tracks/" + ids[1], H)
t = json.loads(b)
aid = t["artists"][0]["id"]
out.append(f"spotify_track keys: {sorted(t.keys())}")
s, b = get("https://api.spotify.com/v1/artists/" + aid, H)
a = json.loads(b)
out.append(f"spotify_artist {a.get('name')}: keys={sorted(a.keys())} genres={a.get('genres')} popularity={a.get('popularity')}")
for l in out:
    print(l)
