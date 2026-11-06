#!/usr/bin/env python3
"""Stil-Tags je Hauptkuenstler aus MusicBrainz (gleiche Abfrage fuer Chart- und Pool-Songs).

Aufruf: python mb_tags.py POOL_TRACKS.csv OUT.csv
Ein Suchaufruf pro Kuenstler (~1/s, Limit von MusicBrainz). Treffer nur bei Score >= 90 und
gleichem normalisierten Namen. Bereits abgefragte Kuenstler aus OUT.csv werden uebersprungen.
"""
import csv, json, os, re, sys, time, unicodedata, urllib.parse, urllib.request

UA = "nmf-top75/1.0 (https://github.com/tillclaude-bit/nmf-top75)"


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main_artist(a):
    return re.split(r",|;| feat\.| ft\.| x | & ", str(a))[0].strip()


def search(name):
    q = urllib.parse.urlencode({"query": f'artist:"{name}"', "fmt": "json", "limit": 5})
    for i in range(5):
        try:
            req = urllib.request.Request("https://musicbrainz.org/ws/2/artist/?" + q, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("artists", [])
        except Exception:
            time.sleep(3 + 3 * i)
    return None


def main():
    pool_f, out = sys.argv[1], sys.argv[2]
    names = {}
    for f, col, grp in ((pool_f, "artists", "pool"), ("data/charts/songs.csv", "artist", "chart")):
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                m = main_artist(r[col])
                if m:
                    names.setdefault(norm(m), {"name": m, "grp": set()})["grp"].add(grp)
    done = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            done = {r["key"]: r for r in csv.DictReader(fh)}
    cols = ["key", "artist", "groups", "mbid", "mb_name", "score", "country", "type", "tags"]
    rows = dict(done)
    n = 0
    for k, v in names.items():
        if k in rows:
            continue
        res = search(v["name"])
        time.sleep(1.1)
        row = {"key": k, "artist": v["name"], "groups": "+".join(sorted(v["grp"])), "mbid": "", "mb_name": "",
               "score": "", "country": "", "type": "", "tags": "" if res is not None else "ERR"}
        for a in res or []:
            if int(a.get("score", 0)) >= 90 and norm(a.get("name", "")) == k:
                tags = sorted(a.get("tags", []), key=lambda t: -t.get("count", 0))
                row.update(mbid=a["id"], mb_name=a["name"], score=a["score"], country=a.get("country", ""),
                           type=a.get("type", ""), tags=";".join(f"{t['name']}:{t.get('count', 0)}" for t in tags))
                break
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
        hit = [r for r in sel if r["mbid"]]
        tg = [r for r in sel if r["tags"] and r["tags"] != "ERR"]
        print(f"{g}: {len(sel)} Kuenstler, Treffer {len(hit)} ({100*len(hit)//max(1,len(sel))}%), mit Tags {len(tg)} ({100*len(tg)//max(1,len(sel))}%)")


if __name__ == "__main__":
    main()
