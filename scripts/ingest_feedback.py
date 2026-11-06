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
        rows.append(dict(track_id=r.track_id, signal="weak_dislike", artist=r.artists, title=r.name,
                         reason="aus Auswahl entfernt", date=date))  # Entfernen = schwaches Negativsignal (Till streicht auch Mittelmaessiges)
# Hat er die Liste deutlich gekuerzt, sind die behaltenen Songs seine Likes
if len(his) < 0.6 * len(mine):
    for r in his[his.track_id.isin(mine.track_id)].itertuples():
        if r.track_id not in known:
            rows.append(dict(track_id=r.track_id, signal="like" if r.pos / len(his) <= 0.6 else "neutral", artist=r.artists, title=r.name,
                             reason=f"behalten, Rang {r.pos} von {len(his)}", date=date))  # grobes Ranking seiner Reihenfolge: obere 60 % like, Rest neutral
            known.add(r.track_id)
# Gestuftes Ranking: nur wenn die Reihenfolge deutlich von meiner abweicht (er hat sortiert)
from scipy.stats import spearmanr
m = his.merge(mine[["track_id", "nr"]], on="track_id")
reordered = len(m) > 5 and abs(spearmanr(m.pos, m.nr)[0]) < 0.95
if reordered:
    n = len(his)
    for r in his.itertuples():
        frac = r.pos / n
        sig = "like" if frac <= 0.45 else ("neutral" if frac <= 0.67 else "weak_dislike")
        if r.track_id in known and not (sig == "weak_dislike"):
            continue
        if r.track_id in known and sig == "weak_dislike":
            continue
        rows.append(dict(track_id=r.track_id, signal=sig, artist=r.artists, title=r.name,
                         reason=f"Rang {r.pos} von {n} in eigener Liste", date=date))
        known.add(r.track_id)
if rows:
    pd.concat([fb, pd.DataFrame(rows)], ignore_index=True).to_csv("data/feedback.csv", index=False)
from collections import Counter
print(f"Feedback {date}: neu {dict(Counter(r['signal'] for r in rows))}, umsortiert={reordered}, Liste hat {len(his)} Songs")
