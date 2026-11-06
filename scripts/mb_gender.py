#!/usr/bin/env python3
"""Geschlecht/Typ der Kuenstler aus MusicBrainz (nur fuer Kuenstler mit bekannter MBID aus mb_tags.csv).

Aufruf: python mb_gender.py OUT.csv   -- inkrementell, ~1 Aufruf/s.
Spalten: key, mbid, type, gender (male/female/non-binary/''; Gruppen haben meist kein Geschlecht)
"""
import csv, json, os, sys, time, urllib.request

UA = "nmf-top75/1.0 (https://github.com/tillclaude-bit/nmf-top75)"
out = sys.argv[1]
mb = [r for r in csv.DictReader(open("data/analysis/mb_tags.csv", encoding="utf-8")) if r["mbid"]]
done = {}
if os.path.exists(out):
    done = {r["key"]: r for r in csv.DictReader(open(out, encoding="utf-8"))}
cols = ["key", "mbid", "type", "gender"]
n = 0
for r in mb:
    if r["key"] in done:
        continue
    row = None
    for i in range(4):
        try:
            req = urllib.request.Request(f"https://musicbrainz.org/ws/2/artist/{r['mbid']}?fmt=json", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as f:
                d = json.load(f)
            row = {"key": r["key"], "mbid": r["mbid"], "type": d.get("type") or "", "gender": (d.get("gender") or "").lower()}
            break
        except Exception:
            time.sleep(3 + 3 * i)
    time.sleep(1.1)
    if row is None:
        continue
    done[r["key"]] = row
    n += 1
    if n % 50 == 0:
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(done.values())
        print(f"{len(done)}/{len(mb)}", flush=True)
with open(out, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(done.values())
g = [r["gender"] for r in done.values()]
print(f"{len(done)} Kuenstler: male {g.count('male')}, female {g.count('female')}, ohne Geschlecht {g.count('')}")
