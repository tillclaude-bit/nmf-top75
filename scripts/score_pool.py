#!/usr/bin/env python3
"""Bewertet den NMF-Pool einer Woche und waehlt die Top 75.

Aufruf: python score_pool.py 2026-10-30

Signale (alle fuer jeden Song gleich berechnet):
  taste   Passungsmass: logistische Regression "deine Chartsongs" gegen "alle NMF-Songs"
          auf eigenen Vorschau-Deskriptoren (Klang, Rhythmus, Timbre; Out-of-fold fuer Pool-Songs)
  breadth log(Anzahl der NMF-Playlists, in denen der Song steht)
  pos     bester Listenplatz (Log)
  artist  Zahl der Chartsongs desselben Kuenstlers in deinen Charts
Gewichte sind vorerst Annahmen; sie werden mit den Chartergebnissen nachjustiert.
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATE = sys.argv[1]
W = {"taste": 1.0, "breadth": 0.5, "pos": 0.25, "artist": 0.5}
MAX_PER_ARTIST = 2
N = 75
QUOTA = 2  # Plaetze je Nicht-Englisch-Markt fuer lokale Songs (in hoechstens 2 NMF-Listen)
MARKETS = ["France", "Deutschland", "Espa", "Italia", "Suomi", "Korea", "Polska", "NL", "Denmark", "Norway", "Sweden", "Greece"]


def norm(s):
    s = re.sub(r"[\(\[].*?[\)\]]", " ", str(s))
    s = re.split(r"\s-\s", s)[0]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def artist_tokens(a):
    return [norm(x) for x in re.split(r",|;|&| feat\.| ft\.| x ", str(a)) if norm(x)]


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")
pool = rd(f"data/pools/pool_tracks_{DATE}.csv")
pdesc = rd(f"data/pools/pool_descriptors_{DATE}.csv")
cdesc = rd("data/analysis/preview_descriptors_2026-10-30.csv")
cdesc = cdesc[cdesc.source == "chart"]
songs = rd("data/charts/songs.csv")
feats = rd("data/charts/song_features.csv")
arts = rd("data/charts/artists.csv")
fb = rd("data/feedback.csv")  # track_id, signal (like/dislike), artist, title, reason, date

DESC = [c for c in pdesc.columns if c not in ("track_id", "source", "preview_ok", "error", "dur")]

# --- Chart-Songs mit Erfolgsgewicht (hoher Peak / lange in den Charts = staerkeres Signal)
cs = feats[["song_id", "spotify_track_id"]].merge(songs[["song_id", "peak_pos", "weeks_in_10_charts"]], on="song_id")
cs["w"] = 1 + 0.5 * (cs.peak_pos <= 10) + 0.5 * (cs.weeks_in_10_charts >= 4)
cd = cdesc.rename(columns={"track_id": "spotify_track_id"}).merge(cs, on="spotify_track_id")
cd = cd[cd.preview_ok == 1]

# --- Pool: Chartsongs und Kuenstler/Titel-Duplikate ausschliessen (keine Eigen-Treffer als Negativ)
pool["tkey"] = pool.artists.map(lambda a: artist_tokens(a)[0] if artist_tokens(a) else "") + "|" + pool.name.map(norm)
chart_keys = set(
    songs.artist.map(lambda a: artist_tokens(a)[0] if artist_tokens(a) else "") + "|" + songs.title.map(norm)
)
pool = pool.merge(pdesc.drop(columns=["source"], errors="ignore"), on="track_id", how="left")
pool["is_chart"] = pool.tkey.isin(chart_keys)
print(f"Pool: {len(pool)} Tracks, davon bereits in deinen Charts: {pool.is_chart.sum()}, "
      f"ohne Deskriptoren: {(pool.preview_ok != 1).sum()}")

# --- Nur Audio-Deskriptoren (gleiche Abdeckung fuer alle Songs; Genre-Labels waren
# bei Chart- und Pool-Songs ungleich verfuegbar und wuerden das Modell verfaelschen)
Xc = cd[DESC].reset_index(drop=True)
Xp = pool[DESC].reset_index(drop=True)
med = pd.concat([Xc, Xp]).median()
Xc, Xp = Xc.fillna(med), Xp.fillna(med)
neg = ~pool.is_chart.values
likes = set(fb[fb.signal == "like"].track_id.astype(str)) if len(fb) else set()
is_like = pool.track_id.isin(likes).values
# Gefallene Songs zaehlen als zusaetzliche Positive (Gewicht 2), nicht als Negative
neg = neg & ~is_like
X = pd.concat([Xc, Xp[is_like], Xp[neg]], ignore_index=True)
y = np.r_[np.ones(len(Xc) + is_like.sum()), np.zeros(neg.sum())]
w = np.r_[cd.w.values, np.full(is_like.sum(), 2.0), np.ones(neg.sum())]
off = len(Xc) + is_like.sum()


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.02, class_weight="balanced", max_iter=3000))


oof = np.zeros(len(X))
for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X, y):
    m = model()
    m.fit(X.iloc[tr], y[tr], logisticregression__sample_weight=w[tr])
    oof[te] = m.decision_function(X.iloc[te])
print(f"Passungsmass CV-AUC (Chartsongs vs. Pool): {roc_auc_score(y, oof):.3f}")
pool["taste"] = np.nan
pool.loc[neg, "taste"] = oof[off:]
# Pool-Songs, die schon in den Charts sind, nur zur Anzeige: mit Gesamtmodell
mfull = model().fit(X, y, logisticregression__sample_weight=w)
pool.loc[~neg, "taste"] = mfull.decision_function(Xp[~neg])

# --- Weitere Signale
pool["breadth"] = np.log(pool.n_playlists)
pool["pos"] = -np.log(pool.best_pos)
hits = {}
for _, r in arts.iterrows():
    for t in artist_tokens(r.artist):
        hits[t] = max(hits.get(t, 0), r.songs_charted)
pool["artist_hits"] = pool.artists.map(lambda a: max([hits.get(t, 0) for t in artist_tokens(a)] or [0]))
pool["artist"] = np.log1p(pool.artist_hits)

pool["score"] = sum(W[k] * z(pool[k]) for k in W)
pool.loc[pool.preview_ok != 1, "score"] -= 0.5  # keine Deskriptoren: vorsichtiger Abschlag

# --- Feedback / Ausschluesse
bad = set(fb[fb.signal == "dislike"].track_id.astype(str)) if len(fb) else set()
pool = pool[~pool.track_id.isin(bad)].reset_index(drop=True)
pool = pool.sort_values("score", ascending=False).reset_index(drop=True)
pool["rank"] = np.arange(1, len(pool) + 1)

# --- Auswahl: keine Chart-Duplikate, max. 2 pro Kuenstler, keine Doppelversionen
# Erst je Nicht-Englisch-Markt die QUOTA besten lokalen Songs (Sprachvielfalt), dann der Rest nach Score
sel, per_artist, seen = [], {}, set()


def take(i, r):
    a = artist_tokens(r.artists)[0] if artist_tokens(r.artists) else r.artists
    if r.is_chart or r.tkey in seen or per_artist.get(a, 0) >= MAX_PER_ARTIST:
        return False
    sel.append(i)
    per_artist[a] = per_artist.get(a, 0) + 1
    seen.add(r.tkey)
    return True


pool["quota_pick"] = False
for m in MARKETS:
    n = 0
    for i, r in pool[(pool.n_playlists <= 2) & pool.playlists.str.contains(m)].iterrows():
        if n >= QUOTA:
            break
        if take(i, r):
            pool.loc[i, "quota_pick"] = True
            n += 1
for i, r in pool.iterrows():
    if len(sel) >= N:
        break
    if i not in sel:
        take(i, r)
pool["selected"] = pool.index.isin(sel)

import os

os.makedirs("data/rankings", exist_ok=True)
keep_cols = ["rank", "selected", "quota_pick", "score", "taste", "breadth", "pos", "artist_hits", "track_id", "name", "artists",
             "n_playlists", "best_pos", "preview_ok", "is_chart", "tempo", "rms_db_mean", "playlists"]
pool[keep_cols].to_csv(f"data/rankings/ranking_{DATE}.csv", index=False)
top = pool[pool.selected].reset_index(drop=True)
top.insert(0, "nr", np.arange(1, len(top) + 1))
top[["nr", "track_id", "name", "artists", "score", "n_playlists", "best_pos"]].to_csv(
    f"data/rankings/playlist_{DATE}.csv", index=False
)
print(f"{len(top)} Songs ausgewaehlt, {top.artists.nunique()} Kuenstler")
print(top[["nr", "name", "artists", "n_playlists", "best_pos", "taste"]].to_string(max_colwidth=38))
