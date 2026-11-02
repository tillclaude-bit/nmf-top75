#!/usr/bin/env python3
"""Loest Tracklisten (Artist - Titel) per Spotify-Suche zu Track-IDs auf und legt optional eine Playlist an.

Fuer jede Datei data/custom/lists/NAME.txt:
  - erste Zeile "# name: Playlist-Name" (sonst NAME), danach je Zeile "Artist - Titel"
  - schreibt data/custom/results/NAME.matches.json (Treffer + nicht gefundene Titel)
  - legt die Playlist nur an, wenn data/custom/lists/NAME.create existiert und noch keine
    data/custom/results/NAME.playlist.json vorhanden ist. Die Playlist ist privat.
Token: auth/refresh_token.enc (mit SPOTIFY_CLIENT_SECRET verschluesselt), wie in create_playlist.py.
"""
import glob, json, os, pathlib, re, subprocess, sys, time, unicodedata
import urllib.error, urllib.parse, urllib.request

cid = os.environ["SPOTIFY_CLIENT_ID"]
rt = subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-a",
                     "-pass", "env:SPOTIFY_CLIENT_SECRET", "-in", "auth/refresh_token.enc"],
                    capture_output=True, text=True, check=True).stdout.strip()


def call(url, data=None, token=None, form=False):
    for attempt in range(5):
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
        req = urllib.request.Request(url, data=body, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = r.read().decode()
                return r.status, (json.loads(txt) if txt else {})
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After", "2")) + 1)
                continue
            return e.code, e.read().decode()[:300]
    return 429, "rate limit"


s, t = call("https://accounts.spotify.com/api/token",
            {"grant_type": "refresh_token", "refresh_token": rt, "client_id": cid}, form=True)
if s != 200:
    sys.exit(f"Token-Refresh fehlgeschlagen: {s} {t}")
tok = t["access_token"]
s, me = call("https://api.spotify.com/v1/me", token=tok)
print("me:", s, me.get("id") if isinstance(me, dict) else me)


def norm(x):
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"\bthe\b", " ", x)
    return re.sub(r"[^a-z0-9]+", "", x)


def parse(line):
    artist, title = re.split(r"\s*[–-]\s+", line.strip(), maxsplit=1)
    feats = []
    for m in re.finditer(r"\((?:feat\.?|ft\.?)\s+([^)]*)\)", artist, flags=re.I):
        feats += re.split(r"\s*(?:&|,| and )\s*", m.group(1))
    main = re.sub(r"\((?:feat\.?|ft\.?)[^)]*\)", "", artist, flags=re.I).strip()
    mains = [a.strip() for a in re.split(r"\s*(?:&|,| and )\s*", main) if a.strip()]
    # Kuenstler mit Komma im Namen (Black Country, New Road) zusammenhalten
    if main.lower().startswith("black country"):
        mains = [main]
    return main, mains, feats, title.strip()


def title_variants(title):
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    stripped = re.sub(r"[.\s]+$", "", stripped) if stripped.endswith(".") else stripped
    out = [title]
    if stripped and stripped != title:
        out.append(stripped)
    return out


BAD = ("live", "karaoke", "instrumental", "acoustic", "sped up", "slowed", "cover", "tribute",
       "remix", "version", "edit", "mix", "reprise", "demo")


def best(line):
    main, mains, feats, title = parse(line)
    wanted_artists = [norm(a) for a in mains + feats if norm(a)]
    variants = title_variants(title)
    cands = {}
    for v in variants:
        for q in (f'track:"{v}" artist:"{mains[0]}"', f"{v} {main}"):
            s, r = call("https://api.spotify.com/v1/search?" + urllib.parse.urlencode(
                {"q": q, "type": "track", "limit": 10, "market": "from_token"}), token=tok)
            if s != 200:
                continue
            for i, it in enumerate(r.get("tracks", {}).get("items", []) or []):
                if it and it.get("id") and it["id"] not in cands:
                    cands[it["id"]] = (len(cands), it)
    scored = []
    for order, it in cands.values():
        names = [norm(a["name"]) for a in it["artists"]]
        art_ok = any(w and (w in n or n in w) for w in wanted_artists[:1] or [] for n in names)
        if not art_ok:
            continue
        tt = norm(it["name"])
        sc = 0
        if tt == norm(variants[0]):
            sc = 6
        elif len(variants) > 1 and tt == norm(variants[1]):
            sc = 5
        elif any(tt.startswith(norm(v)) and norm(v) for v in variants):
            sc = 3
        else:
            continue
        low = it["name"].lower() + " " + (it.get("album", {}).get("name") or "").lower()
        for b in BAD:
            if b in low and b not in line.lower():
                sc -= 2
        extra = sum(1 for w in wanted_artists[1:] if any(w in n or n in w for n in names))
        sc += extra * 0.5
        if it.get("album", {}).get("album_type") == "compilation":
            sc -= 1
        scored.append((sc, -order, it))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    sc, _, it = scored[0]
    if sc <= 0:
        return None
    return {"id": it["id"], "title": it["name"], "artists": [a["name"] for a in it["artists"]],
            "album": it.get("album", {}).get("name"), "score": sc}


pathlib.Path("data/custom/results").mkdir(parents=True, exist_ok=True)
for lst in sorted(glob.glob("data/custom/lists/*.txt")):
    base = pathlib.Path(lst).stem
    resf = pathlib.Path(f"data/custom/results/{base}.matches.json")
    lines = [l.strip() for l in open(lst, encoding="utf-8").read().splitlines() if l.strip()]
    name = base
    if lines and lines[0].startswith("# name:"):
        name = lines[0][len("# name:"):].strip()
        lines = lines[1:]
    if resf.exists():
        matches = json.load(open(resf, encoding="utf-8"))["matches"]
        print(base, "- vorhandene Treffer genutzt:", len(matches))
    else:
        matches = []
        for l in lines:
            m = best(l)
            matches.append({"input": l, "match": m})
            print(("OK  " if m else "MISS"), l, "->", (m["title"] + " / " + ", ".join(m["artists"])) if m else "-")
        json.dump({"name": name, "matches": matches}, open(resf, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    found = [m for m in matches if m["match"]]
    print(f"{base}: {len(found)} von {len(matches)} gefunden")
    plf = pathlib.Path(f"data/custom/results/{base}.playlist.json")
    if os.path.exists(f"data/custom/lists/{base}.create") and not plf.exists():
        ids, seen = [], set()
        for m in found:
            if m["match"]["id"] not in seen:
                seen.add(m["match"]["id"])
                ids.append(m["match"]["id"])
        desc = "Tracklist aus Chat-Vorlage, per Spotify-ID zusammengestellt."
        for url in ("https://api.spotify.com/v1/me/playlists",
                    f"https://api.spotify.com/v1/users/{me.get('id')}/playlists"):
            s, pl = call(url, {"name": name, "public": False, "description": desc}, token=tok)
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
        tot = None
        if s == 200:
            tot = (chk.get("tracks") or chk.get("items") or {}).get("total")
        info = {"name": name, "id": pid, "url": "https://open.spotify.com/playlist/" + pid,
                "tracks_added": len(ids), "tracks_reported": tot, "public": False}
        json.dump(info, open(plf, "w", encoding="utf-8"), indent=1)
        print("PLAYLIST", info)
