#!/usr/bin/env python3
"""Vergleicht einen frischen Abruf (pool_tracks_DATUM_voll.csv) mit dem Pool der Woche (pool_tracks_DATUM.csv).

- Songs, die erst nach dem ersten Abruf in die Listen kamen, landen in pool_tracks_DATUM_nachtrag.csv
  (daraus entsteht die Playlist "Pool - DATUM_nachtrag" zum Export mit Exportify).
- pool_tracks_DATUM.csv wird durch die Vereinigung ersetzt (neue Werte fuer Listenplatz/Breite, alte Songs bleiben);
  der alte Stand wird als pool_tracks_DATUM_v1.csv gesichert.
Aufruf: python pool_nachtrag.py DATUM"""
import sys
import pandas as pd

d = sys.argv[1]
old = pd.read_csv(f"data/pools/pool_tracks_{d}.csv")
new = pd.read_csv(f"data/pools/pool_tracks_{d}_voll.csv")
fresh = new[~new.track_id.isin(old.track_id)]
gone = old[~old.track_id.isin(new.track_id)]
print(f"Erster Abruf: {len(old)} Songs, frischer Abruf: {len(new)} Songs")
print(f"Neu dazugekommen: {len(fresh)} | nur im ersten Abruf (inzwischen aus den Listen): {len(gone)}")
if not len(fresh):
    sys.exit(0)
old.to_csv(f"data/pools/pool_tracks_{d}_v1.csv", index=False)
pd.concat([new, gone], ignore_index=True).to_csv(f"data/pools/pool_tracks_{d}.csv", index=False)
fresh.to_csv(f"data/pools/pool_tracks_{d}_nachtrag.csv", index=False)
print(fresh[["name", "artists", "n_playlists", "best_pos"]].to_string(index=False))
