#!/usr/bin/env python3
"""Liest die zuletzt angelegte Playlist aus Tills Konto und leitet Feedback ab:
 - Song von ihm ergaenzt (nicht in meiner Auswahl)  -> like
 - Song von mir ausgewaehlt, von ihm entfernt        -> dislike (schwaches Signal, nur Ausschluss)
Reihenfolge (sein Ranking) wird archiviert. Idempotent: schon vorhandene track_ids werden nicht doppelt eingetragen.
Aufruf: python ingest_feedback.py   (nutzt die letzte Zeile von data/playlist_ids.csv)"""
import csv, os, subprocess, sys
import pandas as pd

ids = pd.read_csv("data/playlist_ids.csv")
if ids.empty:
    sys.exit("keine Playlist vorhanden")
date, pid = ids.iloc[-1].date, ids.iloc[-1].playlist_id
out = f"data/feedback/user_playlist_{date}.csv"
subprocess.run([sys.executable, "scripts/read_playlist.py", pid, out], check=True)
mine = pd.read_csv(f"data/rankings/playlist_{date}.csv")
his = pd.read_csv(out)
fb = pd.read_csv("data/feedback.csv")
known = set(fb.track_id.astype(str))
rows = []
for r in his[~his.track_id.isin(mine.track_id)].itertuples():
    if r.track_id not in known:
        rows.append(dict(track_id=r.track_id, signal="like", artist=r.artists, title=r.name,
                         reason=f"ergaenzt, Rang {r.pos} in eigener Liste", date=date))
for r in mine[~mine.track_id.isin(his.track_id)].itertuples():
    if r.track_id not in known:
        rows.append(dict(track_id=r.track_id, signal="dislike", artist=r.artists, title=r.name,
                         reason="aus Auswahl entfernt", date=date))
if rows:
    pd.concat([fb, pd.DataFrame(rows)], ignore_index=True).to_csv("data/feedback.csv", index=False)
print(f"Feedback {date}: {sum(r['signal']=='like' for r in rows)} neue likes, "
      f"{sum(r['signal']=='dislike' for r in rows)} neue dislikes, Liste hat {len(his)} Songs")
