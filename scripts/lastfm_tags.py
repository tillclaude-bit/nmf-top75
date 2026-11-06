#!/usr/bin/env python3
"""Stil-Tags und aehnliche Kuenstler je Hauptkuenstler aus Last.fm (gleiche Abfrage fuer Chart- und Pool-Songs).

Aufruf: python lastfm_tags.py POOL_TRACKS.csv OUT.csv   (Schluessel in Umgebungsvariable LASTFM_API_KEY)
Pro Kuenstler: artist.getTopTags (Tag-Gewichte 0-100) und artist.getInfo (Hoererzahl, aehnliche Kuenstler).
Wenn MusicBrainz-ID bekannt (data/analysis/mb_tags.csv), wird sie fuer eindeutige Treffer genutzt.
Bereits abgefragte Kuenstler aus OUT.csv werden uebersprungen.
"""
import csv, json, os, re, sys, time, unicodedata, urllib.parse, urllib.request

KEY = os.environ.get("LASTFM_API_KEY", "")
API = "https://ws.audioscrobbler.com/2.0/"
JUNK = {"seen live", "favorites", "favourites", "favorite", "love", "loved", "female vocalists", "male vocalists",
        "female vocalist", "male vocalist", "singer-songwriter", "00s", "10s", "20s", "90s", "80s", "70s"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main_artist(a):
    return re.split(r",|;| feat\.| ft\.| x | & ", str(a))[0].strip()


def call(method, ident):
    p = {"method": method, "api_key": KEY, "format": "json", "autocorrect": 1}
    p.update(ident)
    for i in range(4):
        try:
            with urllib.request.urlopen(API + "?" + urllib.parse.urlencode(p), timeout=30) as r:
                d = json.load(r)
            if "error" in d:
                return {"_err": d["error"]} if d["error"] in (6, 7) else None
            return d
        except Exception:
            time.sleep(2 + 2 * i)
    return None


def main():
    if not KEY:
        sys.exit("LASTFM_API_KEY fehlt")
    pool_f, out = sys.argv[1], sys.argv[2]
    names = {}
    for f, col, grp in ((pool_f, "artists", "pool"), ("data/charts/songs.csv", "artist", "chart")):
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                m = main_artist(r[col])
                if m:
                    names.setdefault(norm(m), {"name": m, "grp": set()})["grp"].add(grp)
    mbid = {}
    if os.path.exists("data/analysis/mb_tags.csv"):
        with open("data/analysis/mb_tags.csv", encoding="utf-8") as fh:
            mbid = {r["key"]: r["mbid"] for r in csv.DictReader(fh) if r["mbid"]}
    done = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            done = {r["key"]: r for r in csv.DictReader(fh)}
    cols = ["key", "artist", "groups", "lfm_name", "listeners", "tags", "similar"]
    rows = dict(done)
    n = 0
    for k, v in names.items():
        if k in rows and rows[k]["tags"] != "ERR":
            rows[k]["groups"] = "+".join(sorted(set(rows[k]["groups"].split("+")) | v["grp"]))
            continue
        ident = {"mbid": mbid[k]} if k in mbid else {"artist": v["name"]}
        t = call("artist.getTopTags", ident)
        if t is not None and "_err" in t and k in mbid:  # MBID unbekannt -> per Name
            ident = {"artist": v["name"]}
            t = call("artist.getTopTags", ident)
        info = call("artist.getInfo", ident)
        time.sleep(0.3)
        row = {"key": k, "artist": v["name"], "groups": "+".join(sorted(v["grp"])), "lfm_name": "", "listeners": "",
               "tags": "ERR" if t is None else "", "similar": ""}
        if t and "toptags" in t:
            tg = [(x["name"].lower(), int(x.get("count", 0))) for x in t["toptags"].get("tag", [])
                  if x["name"].lower() not in JUNK and int(x.get("count", 0)) >= 10]
            row["tags"] = ";".join(f"{a}:{b}" for a, b in tg[:10])
        if info and "artist" in info:
            a = info["artist"]
            row["lfm_name"] = a.get("name", "")
            row["listeners"] = a.get("stats", {}).get("listeners", "")
            sim = a.get("similar", {}).get("artist", [])
            row["similar"] = ";".join(x["name"] for x in sim[:5])
        rows[k] = row
        n += 1
        if n % 50 == 0:
            with open(out, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows.values())
            print(f"{len(rows)}/{len(names)} abgefragt", flush=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows.values())
    for g in ("pool", "chart"):
        sel = [r for r in rows.values() if g in r["groups"].split("+")]
        tg = [r for r in sel if r["tags"] and r["tags"] != "ERR"]
        er = [r for r in sel if r["tags"] == "ERR"]
        print(f"{g}: {len(sel)} Kuenstler, mit Tags {len(tg)} ({100*len(tg)//max(1,len(sel))}%), Fehler {len(er)}")


if __name__ == "__main__":
    main()
