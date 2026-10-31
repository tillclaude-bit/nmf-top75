#!/usr/bin/env python3
"""Prueft, ob die NMF-Playlists neu sind: Vergleich mit dem Pool der letzten archivierten Woche.
Exit 0 = frisch, Exit 1 = mindestens 3 Playlists fast unveraendert (vermutlich noch nicht aktualisiert).
"""
import csv, glob, re, sys
from collections import defaultdict

date = sys.argv[1]
prev = sorted(p for p in glob.glob("data/pools/pool_raw_*.csv") if re.search(r"(\d{4}-\d{2}-\d{2})", p).group(1) < date)
if not prev:
    print("Kein Vorwochen-Pool zum Vergleich, Pruefung uebersprungen")
    sys.exit(0)


def load(p):
    d = defaultdict(set)
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        d[r["playlistName"]].add(r["trackId"])
    return d


old, new = load(prev[-1]), load(f"data/pools/pool_raw_{date}.csv")
stale = []
for name, ids in new.items():
    o = old.get(name)
    if o and len(ids & o) / max(len(ids), 1) > 0.5:
        stale.append(name)
print(f"Vergleich mit {prev[-1]}: {len(stale)} von {len(new)} Playlists noch unveraendert: {stale}")
sys.exit(1 if len(stale) >= 3 else 0)
