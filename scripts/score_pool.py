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
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
W = {"feat": 0.0, "breadth": 0.0, "pos": 0.25, "artist": 0.5, "stylefit": 0.5}  # style: Stil-Tags der Kuenstler (MusicBrainz) gegen die Stile der Chart-Kuenstler; ohne Tags neutral (0)  # breadth bleibt drin (Till, 2026-10-31); Vorwochen-Backtest: Breite sagt Chart-Einstiege vorher
MAX_PER_ARTIST = 1  # Till, 2026-11-06: kein Hauptkuenstler doppelt; Feature-Gaeste zaehlen nicht
N = 75
# Weihnachtslieder sind ausgeschlossen (Titel-Stichworte in den NMF-Sprachen; Kuenstlernamen zaehlen nicht)
XMAS = re.compile(r"christmas|xmas|x-mas|weihnacht|\bnoel\b|\bnoël\b|kerst|joulu|\bjul\b|julen|jul[ei]n?\b|navidad|natale|\bnatal\b|\bsanta\b|jingle|sleigh|rudolph|mistletoe|silent night|stille nacht|white christmas|boże narodzenie|święta|jõulu|\bjule\b|nadal|nativity|bethlehem|reindeer|snowman|frosty|holly jolly|\bcarols?\b|noche buena|nochebuena|krismas|božić|vánoc", re.I)
LANG_MIN_PROB = 0.40   # darunter gilt die Spracherkennung als unsicher (z.B. Instrumentals)
CHART_WEEKS = 4        # Sprachquote richtet sich nach dem Anteil in den letzten Chartwochen
LANGS = ["en", "de", "fr", "es", "it", "el", "da", "no", "sv", "fi", "ko", "pl", "nl", "pt", "ja", "tr", "ar", "hi", "id"]  # plausible Sprachen
MARKET_LANG = {"UK & Ireland": "en", "AU & NZ": "en", "France": "fr", "Deutschland": "de", "Espa": "es", "Italia": "it",
               "Greece": "el", "Denmark": "da", "Norway": "no", "Sweden": "sv", "Suomi": "fi", "Korea": "ko",
               "Polska": "pl", "NL": "nl"}
MAX_LANG_SHARE = 0.25  # hoechstens dieser Anteil der Nicht-Englisch-Plaetze fuer eine einzelne Sprache


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
songs = rd("data/charts/songs.csv")
feats = rd("data/charts/song_features.csv")
arts = rd("data/charts/artists.csv")
fb = rd("data/feedback.csv")  # track_id, signal (like/dislike), artist, title, reason, date
import glob

# Der Klang-Check aus eigenen 30-s-Vorschauen (librosa-Deskriptoren) ist entfernt (Till, 2026-11-06): aufwendig und schwaecher als die Spotify-Tags/-Features.
# Klang kommt nur noch aus den Spotify-Features des Exportify-Exports (Abschnitt "Echte Spotify-Features").

# --- Pool: Chartsongs und Kuenstler/Titel-Duplikate ausschliessen (keine Eigen-Treffer als Negativ)
pool["tkey"] = pool.artists.map(lambda a: artist_tokens(a)[0] if artist_tokens(a) else "") + "|" + pool.name.map(norm)
chart_keys = set(
    songs.artist.map(lambda a: artist_tokens(a)[0] if artist_tokens(a) else "") + "|" + songs.title.map(norm)
)
pool["is_chart"] = pool.tkey.isin(chart_keys)
pool["preview_ok"] = 0
print(f"Pool: {len(pool)} Tracks, davon bereits in deinen Charts: {pool.is_chart.sum()}")

neg = ~pool.is_chart.values
likes = (set(fb[fb.signal == "like"].track_id.astype(str)) if len(fb) else set()) - set(feats.spotify_track_id.dropna().astype(str))  # Chartsongs sind schon Positive
is_like = pool.track_id.isin(likes).values
neutral = set(fb[fb.signal == "neutral"].track_id.astype(str)) if len(fb) else set()
weak = set(fb[fb.signal == "weak_dislike"].track_id.astype(str)) if len(fb) else set()
neg = neg & ~is_like & ~pool.track_id.isin(neutral).values  # neutral: weder Positiv noch Negativ
pool["taste"] = np.nan

# --- Echte Spotify-Features aus Exportify-CSVs (data/exportify/*.csv und data/feedback/user_playlist_*.csv)
# Positive: deine Top-100 (Exportify-Export, Gewicht 1) und gefallene Songs (Gewicht 2); Negative: Pool-Songs (Dislikes/schwache Dislikes Gewicht 2).
# Ohne Export dieser Woche bleibt das Signal aus, und der Klang-Deskriptor-Geschmackswert zaehlt voll (siehe W unten).
FEAT_COLS = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Mode"]


def _load_exp(files):
    out = []
    for f in files:
        d = rd(f)
        if "Danceability" in d.columns:
            d["tid"] = d["Track URI"].str.split(":").str[-1]
            out.append(d)
    return pd.concat(out, ignore_index=True).drop_duplicates("tid", keep="last") if out else pd.DataFrame(columns=["tid"] + FEAT_COLS)


_exp_all = _load_exp(sorted(glob.glob("data/exportify/*.csv")) + sorted(glob.glob("data/feedback/user_playlist_*.csv")))
_top = _load_exp(sorted(glob.glob("data/exportify/top100_*.csv")))
pool["genres_sp"] = pool.track_id.astype(str).map(dict(zip(_exp_all.tid, _exp_all.get("Genres", pd.Series(dtype=str)).fillna(""))) if len(_exp_all) else {}).fillna("")
pool["feat"] = np.nan
HAVE_FEAT = False
_fe = _exp_all.set_index("tid")[FEAT_COLS].dropna() if len(_exp_all) else pd.DataFrame(columns=FEAT_COLS)  # Songs ohne Klangwerte im Export (kommt vor) werden nicht fuers Modell genutzt
_has = pool.track_id.astype(str).isin(_fe.index).values
_negf = neg & _has
if len(_top) >= 30 and _negf.sum() >= 100:
    _dis = set(fb[fb.signal.isin(["dislike", "weak_dislike"])].track_id.astype(str)) if len(fb) else set()
    _pos_ids = [t for t in dict.fromkeys(_top.tid) if t in _fe.index]
    # Chart-Verlauf der Neueinsteiger der Vorwoche (Till, 2026-11-10): Songs, die vor einer Woche neu eingestiegen sind, zaehlen staerker,
    # wenn sie seither geklettert sind (bis 2,0), und schwaecher, wenn sie gefallen sind (bis 0,75). Alle anderen Chartsongs: 1,0.
    _mom = {}
    _cr0 = rd("data/charts/chart_rows.csv")
    _wk = sorted(_cr0.chart_date.unique())
    if len(_wk) >= 2:
        _a = _cr0[_cr0.chart_date == _wk[-2]]
        _a = _a[_a.prev_pos_shown.isna()]
        _b = _cr0[_cr0.chart_date == _wk[-1]].set_index("song_id").pos
        _s2t = dict(zip(feats.song_id, feats.spotify_track_id.astype(str)))
        for _r in _a.itertuples():
            _now = _b.get(_r.song_id, np.nan)
            _d = (_r.pos - _now) if _now == _now else -25.0  # inzwischen aus den Charts: wie ein deutlicher Absturz
            _mom[_s2t.get(_r.song_id, "")] = 1.0 + float(np.clip(_d / 20.0, -0.25, 1.0))
        print(f"Chart-Verlauf: {len(_mom)} Neueinsteiger von {_wk[-2]}, Gewicht {min(_mom.values()):.2f}..{max(_mom.values()):.2f}, davon gestiegen {sum(v > 1 for v in _mom.values())}")
    _pos_w = [_mom.get(t, 1.0) for t in _pos_ids]
    for t in sorted(likes):
        if t in _fe.index and t not in _pos_ids:
            _pos_ids.append(t)
            _pos_w.append(2.0)
    _pool_ids = pool.track_id.astype(str)[_negf].tolist()
    _old_neg = [t for t in sorted(_dis) if t in _fe.index and t not in set(_pool_ids) and t not in set(_pos_ids)]
    _Xf = pd.concat([_fe.loc[_pos_ids], _fe.loc[_pool_ids], _fe.loc[_old_neg]], ignore_index=True)
    _yf = np.r_[np.ones(len(_pos_ids)), np.zeros(len(_pool_ids) + len(_old_neg))]
    _wf = np.r_[_pos_w, [2.0 if t in _dis else 1.0 for t in _pool_ids], np.full(len(_old_neg), 2.0)]

    def _fmodel():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))

    _oof = np.zeros(len(_Xf))
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(_Xf, _yf):
        m_ = _fmodel()
        m_.fit(_Xf.iloc[tr], _yf[tr], logisticregression__sample_weight=_wf[tr])
        _oof[te] = m_.decision_function(_Xf.iloc[te])
    print(f"Spotify-Features: {len(_pos_ids)} Positive, {len(_pool_ids) + len(_old_neg)} Negative, CV-AUC {roc_auc_score(_yf, _oof):.3f}")
    _mf = _fmodel().fit(_Xf, _yf, logisticregression__sample_weight=_wf)
    _allf = _fe.reindex(pool.track_id.astype(str)).reset_index(drop=True)
    _okf = _allf.notna().all(axis=1).values
    pool.loc[_okf, "feat"] = _mf.decision_function(_allf[_okf])
    _idx = np.where(_negf)[0]
    pool.loc[_idx, "feat"] = _oof[len(_pos_ids):len(_pos_ids) + len(_pool_ids)]  # Out-of-fold fuer die Negativ-Klasse
    HAVE_FEAT = True
    print(f"Songs mit Spotify-Features: {pool.feat.notna().sum()} von {len(pool)}")
else:
    print("Keine Exportify-Daten: Geschmackswert nur aus eigenen Vorschau-Deskriptoren")
if HAVE_FEAT:
    W.update({"feat": 0.5, "stylefit": 1.0})  # 2026-11-13: Klangwerte trennten Keeps/Overlooks kaum (AUC ~0,5) -> halbes Gewicht, Stilpassung dafuer voll
else:
    W.update({"stylefit": 1.0})  # ohne Exportify-Features: Stil-Passung (Last.fm) traegt den Geschmack

# --- Weitere Signale
pool["breadth"] = np.log(pool.n_playlists)
pool["pos"] = -np.log(pool.best_pos)
hits = {}
for _, r in arts.iterrows():
    for t in artist_tokens(r.artist):
        hits[t] = max(hits.get(t, 0), r.songs_charted)
pool["artist_hits"] = pool.artists.map(lambda a: max([hits.get(t, 0) for t in artist_tokens(a)] or [0]))
pool["artist"] = np.log1p(pool.artist_hits)

# --- Stil-Regeln (data/style_rules.csv, von Till): Ausschluss, wenn ein MusicBrainz-Stil-Tag des Hauptkuenstlers den Begriff enthaelt
import os as _os0, unicodedata as _ud
def _nk(a):
    m = re.split(r",|;| feat\.| ft\.| x | & ", str(a))[0]
    m = _ud.normalize("NFKD", m).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", m).strip()
pool["style_tags"] = ""
if _os0.path.exists("data/analysis/mb_tags.csv"):
    _mb = rd("data/analysis/mb_tags.csv").fillna("")
    _tg = dict(zip(_mb.key.astype(str), _mb.tags))
    _ct = dict(zip(_mb.key.astype(str), _mb.country))
    # Namensverwechslungen vermeiden: Herkunftsland des MusicBrainz-Kuenstlers muss zu den NMF-Laendern des Songs passen
    _EN = {"GB", "IE", "US", "CA", "AU", "NZ"}
    _MC = {"UK & Ireland": _EN, "AU & NZ": _EN, "France": {"FR", "BE", "CH", "CA"}, "Deutschland": {"DE", "AT", "CH"}, "Espa": {"ES", "MX", "AR", "CO", "CL", "PR", "US"},
           "Italia": {"IT", "CH"}, "Greece": {"GR", "CY"}, "Denmark": {"DK"}, "Norway": {"NO"}, "Sweden": {"SE"}, "Suomi": {"FI"},
           "Korea": {"KR"}, "Polska": {"PL"}, "NL": {"NL", "BE"}}
    def _tags_ok(r):
        k = _nk(r.artists)
        c = _ct.get(k, "")
        if not c:
            return _tg.get(k, "")
        allowed = set().union(*[v for m, v in _MC.items() if m in r.playlists]) if any(m in r.playlists for m in _MC) else set()
        return _tg.get(k, "") if c in allowed else ""
    pool["style_tags"] = pool.apply(_tags_ok, axis=1)
# --- Stil-Passung: Stil-Tags (MusicBrainz) der Kuenstler; Logistische Regression "Chart-Kuenstler" gegen "uebrige NMF-Kuenstler", out-of-fold
import os as _os1
pool["stylefit"] = np.nan
if _os1.path.exists("data/analysis/mb_tags.csv"):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import cross_val_predict
    _m = rd("data/analysis/mb_tags.csv").fillna("")
    _m = _m[(_m.tags != "") & (_m.tags != "ERR")].copy()
    def _doc(t, n=6):
        out = []
        for x in [y for y in t.split(";") if y][:n]:
            nm, c = x.rsplit(":", 1)
            if int(c) >= 1:
                out.append(re.sub(r"[^a-z0-9]+", "_", nm.lower().replace("-", " ")))
        return " ".join(out)
    _m["doc"] = _m.tags.map(_doc)
    _m = _m[_m.doc != ""].reset_index(drop=True)
    _m["y"] = _m.groups.str.contains("chart").astype(int)
    if _m.y.sum() >= 20 and (1 - _m.y).sum() >= 20:
        _cv = CountVectorizer(token_pattern=r"\S+", binary=True, min_df=2).fit(_m.doc)
        _X = _cv.transform(_m.doc)
        _clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
        _m["fit"] = cross_val_predict(_clf, _X, _m.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
        print(f"Stil-Passung CV-AUC (Chart-Kuenstler vs. Rest): {roc_auc_score(_m.y, _m.fit):.3f} ({len(_m)} Kuenstler mit Tags)")
        _fit = dict(zip(_m.key.astype(str), _m.fit))
        pool["stylefit"] = [(_fit.get(_nk(a)) if t else np.nan) for a, t in zip(pool.artists, pool.style_tags)]
        print(f"Songs mit Stil-Passung: {pool.stylefit.notna().sum()} von {len(pool)}")

# --- Stil-Tags aus Last.fm (gleiche Abfrage fuer alle Kuenstler): ueberschreibt die MusicBrainz-Stil-Passung, wenn vorhanden
pool["lf_tags"] = ""
_mb_fit = pool["stylefit"].copy()  # MusicBrainz-Stilpassung als Rueckfall, wenn Last.fm den Kuenstler nicht kennt
if _os1.path.exists("data/analysis/lastfm_tags.csv"):
    from sklearn.feature_extraction.text import CountVectorizer as _CV2
    from sklearn.model_selection import cross_val_predict as _cvp2
    _lf = rd("data/analysis/lastfm_tags.csv").fillna("")
    # Namensverwechslungen (z.B. "Oria" -> "Orla") ausschliessen: Last.fm-Name muss dem Kuenstlernamen entsprechen
    _lf["ok"] = [(_nk(n) == k or n == "") and t not in ("", "ERR") for n, k, t in zip(_lf.lfm_name, _lf.key, _lf.tags)]
    _lf = _lf[_lf.ok].copy()
    _lf["doc"] = _lf.tags.map(lambda t: " ".join(re.sub(r"[^a-z0-9]+", "_", x.rsplit(":", 1)[0]) for x in [y for y in t.split(";") if y][:8] if int(x.rsplit(":", 1)[1]) >= 10))
    _lf = _lf[_lf.doc != ""].reset_index(drop=True)
    _lf["y"] = _lf.groups.str.contains("chart").astype(int)
    if _lf.y.sum() >= 20 and (1 - _lf.y).sum() >= 20:
        _cv2 = _CV2(token_pattern=r"\S+", binary=True, min_df=2).fit(_lf.doc)
        _X2 = _cv2.transform(_lf.doc)
        _clf2 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        _lf["fit"] = _cvp2(_clf2, _X2, _lf.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
        print(f"Stil-Passung Last.fm CV-AUC (Chart-Kuenstler vs. Rest): {roc_auc_score(_lf.y, _lf.fit):.3f} ({len(_lf)} Kuenstler mit Tags)")
        _fit2 = dict(zip(_lf.key.astype(str), _lf.fit))
        pool["stylefit"] = [_fit2.get(_nk(a), np.nan) for a in pool.artists]
        pool["lf_tags"] = [dict(zip(_lf.key.astype(str), _lf.tags)).get(_nk(a), "") for a in pool.artists]
        pool["stylefit"] = pool["stylefit"].fillna(_mb_fit)
        print(f"Songs mit Stil-Passung (Last.fm): {pool.stylefit.notna().sum()} von {len(pool)}")

_zs = {k: np.nan_to_num(z(pool[k]), nan=0.0) for k in W}  # fehlende Werte (z.B. keine Tags) zaehlen neutral
pool["score"] = sum(W[k] * _zs[k] for k in W)
pool.loc[pool.feat.isna() & HAVE_FEAT, "score"] -= 0.5  # Features erwartet, aber fuer diesen Song nicht im Export: vorsichtiger Abschlag

# --- Feedback / Ausschluesse
bad = set(fb[fb.signal == "dislike"].track_id.astype(str)) if len(fb) else set()
if len(fb) and "date" in fb.columns:  # Neuberechnung einer schon bewerteten Woche: bereits entfernte Songs (schwache Dislikes) nicht erneut vorschlagen
    bad |= set(fb[(fb.signal == "weak_dislike") & (fb.date.astype(str) == DATE)].track_id.astype(str))
# Recherche-Urteile (data/research/verdicts_DATUM.csv, von der woechentlichen Song-Recherche): verdict "no" schliesst aus, "ok" gibt einen kleinen Bonus
import os as _os2
_vf = f"data/research/verdicts_{DATE}.csv"
verdict = {}
verdict_voice = {}
if _os2.path.exists(_vf):
    _v = rd(_vf).fillna("")
    verdict = dict(zip(_v.track_id.astype(str), _v.verdict.str.lower().str.strip()))
    if "voice" in _v.columns:
        verdict_voice = dict(zip(_v.track_id.astype(str), _v.voice.str.lower().str.strip()))
    bad |= {t for t, v in verdict.items() if v == "no"}
    print(f"Recherche-Urteile: {len(verdict)} (no: {sum(v == 'no' for v in verdict.values())}, ok: {sum(v == 'ok' for v in verdict.values())}, unsure: {sum(v == 'unsure' for v in verdict.values())})")
else:
    print("Recherche-Urteile: keine Datei, Auswahl ohne Recherche")
# Ausschluss gilt fuer alle Spotify-Versionen desselben Songs (gleicher Kuenstler|Titel), nicht nur fuer die markierte Track-ID
bad_keys = set(pool[pool.track_id.astype(str).isin(bad)].tkey)
pool = pool[~(pool.track_id.astype(str).isin(bad) | pool.tkey.isin(bad_keys))].reset_index(drop=True)

pool["style_excl"] = ""
pool["style_flag"] = ""
y_c = lambda x: x.rsplit(":", 1)[1] if ":" in x else "0"
if _os0.path.exists("data/style_rules.csv"):
    for _, _r in rd("data/style_rules.csv").iterrows():
        if _r.effect not in ("exclude", "penalty"):
            continue
        _t = _r.term.lower()
        _contains = ("match" in _r.index) and str(_r.match) == "contains"
        def _m(tag):
            tag = tag.split(":")[0].strip().lower().replace("-", " ")
            return (_t.replace("-", " ") in tag) if _contains else (tag == _t.replace("-", " "))
        # Spotify-Genres des Kuenstlers (Exportify) sind massgeblich: sie sind je Kuenstler eindeutig (keine Namensverwechslung).
        # Ohne Spotify-Genres entscheidet MusicBrainz (mit Laender-Check). Last.fm allein (Namensabgleich ist unsicher) schliesst nicht aus,
        # sondern gibt Abzug und Markierung "style_flag", die die woechentliche Recherche pruefen muss.
        sp_has = pool.genres_sp != ""
        sp_hit = pool.genres_sp.apply(lambda g: any(_m(x) for x in g.split(",") if x))
        mb_hit = pool.style_tags.apply(lambda t: any(_m(x) for x in [y for y in t.split(";") if y][:2]))  # nur die 2 haeufigsten Tags
        lf_hit = pool.lf_tags.apply(lambda t: any(_m(x) for x in [y for y in t.split(";") if y][:2] if int(y_c(x)) >= 30))
        hit = np.where(sp_has, sp_hit, mb_hit)
        hit = pd.Series(hit, index=pool.index).astype(bool)
        if _r.effect == "penalty":  # weicher Stil: Abzug statt Ausschluss (nur Spotify-Genres bzw. MusicBrainz)
            pool.loc[hit, "score"] -= 1.5
            print(f"Stil-Abzug '{_r.term}': {int(hit.sum())} Songs")
            continue
        flag = lf_hit & ~sp_has & ~hit
        pool.loc[flag & (pool.style_flag == ""), "style_flag"] = _r.term
        pool.loc[hit & (pool.style_excl == ""), "style_excl"] = _r.term
    _ex = pool[pool.style_excl != ""]
    print(f"Stil-Ausschluesse: {len(_ex)}")
    _ex[["track_id", "name", "artists", "style_excl", "style_tags", "lf_tags", "genres_sp", "score"]].to_csv(f"data/rankings/style_excluded_{DATE}{SUFFIX}.csv", index=False)
    pool = pool[pool.style_excl == ""].reset_index(drop=True)
pool.loc[pool.style_flag != "", "score"] -= 1.5  # nur Last.fm meldet einen unerwuenschten Stil: Abzug, Recherche prueft
pool["score"] = pool.score + 0.5 * pool.track_id.astype(str).map(lambda t: 1.0 if verdict.get(t) == "ok" else 0.0)
pool = pool.sort_values("score", ascending=False).reset_index(drop=True)
pool["rank"] = np.arange(1, len(pool) + 1)

# --- Sprache (Whisper auf den Vorschauen) und Zielanteil aus den aktuellen Charts
import glob as _glob
lang = None
lf = sorted(_glob.glob("data/analysis/language_*.csv"))
if lf:
    lang = pd.concat([rd(f) for f in lf], ignore_index=True)
    lang = lang[lang.lang.notna() & (lang.lang != "")].drop_duplicates("track_id", keep="last")
    lang["lang_ok"] = np.where((lang.prob >= LANG_MIN_PROB) & lang.lang.isin(LANGS), lang.lang, "unknown")
    pool = pool.merge(lang[["track_id", "lang", "prob", "lang2", "prob2"]].rename(
        columns={"lang": "w1", "prob": "p1", "lang2": "w2", "prob2": "p2"}), on="track_id", how="left")

    def pick(r):
        """Whisper-Top-2, gewichtet mit der Herkunft: Sprachen der Laender, in deren NMF-Liste der Song steht, x4"""
        if pd.isna(r.w1):
            return "unknown"
        home = {l for k, l in MARKET_LANG.items() if k in r.playlists}
        c = {r.w1: r.p1}
        if isinstance(r.w2, str) and r.w2:
            c[r.w2] = r.p2
        b = {l: p * (4.0 if l in home else 1.0) for l, p in c.items()}
        best = max(b, key=b.get)
        conf = b[best] / (sum(b.values()) + max(0.0, 1 - sum(c.values())))
        return best if (conf >= LANG_MIN_PROB and best in LANGS) else "unknown"

    pool["lang"] = pool.apply(pick, axis=1)
    import os as _os
    if _os.path.exists("data/language_overrides.csv"):  # von Hand bestaetigte Sprachen haben Vorrang
        ov = rd("data/language_overrides.csv").set_index("track_id").lang.to_dict()
        pool["lang"] = [ov.get(t, l) for t, l in zip(pool.track_id, pool.lang)]
    cr = rd("data/charts/chart_rows.csv").merge(
        feats[["song_id", "spotify_track_id"]], on="song_id", how="left"
    ).merge(lang[["track_id", "lang_ok"]].rename(columns={"track_id": "spotify_track_id"}), on="spotify_track_id", how="left")
    weeks = sorted(cr.chart_date.unique())[-CHART_WEEKS:]
    cl = cr[cr.chart_date.isin(weeks) & cr.lang_ok.notna() & (cr.lang_ok != "unknown")]
    EN_SHARE = float((cl.lang_ok == "en").mean())
    print(f"Chartwochen {weeks[0]}..{weeks[-1]}: Anteil Englisch {EN_SHARE:.0%} ({len(cl)} Chartplaetze mit Spracherkennung)")
else:
    pool["lang"] = "unknown"
    EN_SHARE = None
    print("Keine Sprachdaten, Auswahl ohne Sprachquote")

# --- Stimme: maennliche Stimmen sollen seltener vorkommen als weibliche (Till, 2026-11-06): hoechstens 15 % unter den Songs mit bekannter Stimme.
# Quelle: Web-Recherche der woechentlichen Song-Recherche (Spalte voice in verdicts_DATUM.csv: female | male | mixed; bei Bands Geschlecht der Leadstimme, mixed bei
# gemischter Leadstimme); ohne Urteil Rueckfall auf MusicBrainz (nur Einzelpersonen). Umsetzung als kleinster Score-Abzug fuer maennliche Stimmen
# (0 bis 3,0 in 0,25-Schritten), mit dem der Hoechstanteil in der Auswahl erreicht wird.
MALE_MAX_SHARE = 0.15
pool["voice"] = "unknown"
PEN = 0.0
if _os1.path.exists("data/analysis/mb_gender.csv"):
    _gd = rd("data/analysis/mb_gender.csv").fillna("")
    _vm = {r.key: ("male" if r.gender == "male" else "female" if r.gender == "female" else "unknown") for r in _gd.itertuples()}
    pool["voice"] = [_vm.get(_nk(a), "unknown") for a in pool.artists]
pool["voice_src"] = np.where(pool.voice != "unknown", "mb", "")
_wv = pool.track_id.astype(str).map(verdict_voice).fillna("")
_wv = _wv.where(_wv.isin(["male", "female", "mixed"]), "")
pool.loc[_wv != "", "voice"] = _wv[_wv != ""]
pool.loc[_wv != "", "voice_src"] = "web"
print("Stimmen im Pool (Songs):", pool.voice.value_counts().to_dict(), "| Quelle Web-Recherche:", int((pool.voice_src == "web").sum()))
MALE_CAP = MALE_MAX_SHARE

# --- Auswahl: keine Chart-Duplikate, max. 2 pro Kuenstler, keine Doppelversionen
# Erst die besten nicht-englischen Songs bis zum Zielanteil (aus den aktuellen Charts), dann der Rest nach Score
def _select(N):
    sel, per_artist, seen = [], {}, set()


    TASTE_MIN = -1e9  # Schwelle ausgeschaltet (Till, 2026-11-06: Passungsmass unter 0 schliesst nicht automatisch aus; Klang-Deskriptoren sind schwach). Vorher 0.0
    # (ohne Vorschau gibt es kein Passungsmass: diese Songs sind vom Filter ausgenommen)


    def take(i, r, strict=True):
        if r.is_xmas:
            return False
        toks = (artist_tokens(r.artists) or [str(r.artists)])[:1]  # nur der Hauptkuenstler (erster genannter) zaehlt
        if r.is_chart or r.tkey in seen or any(per_artist.get(t, 0) >= MAX_PER_ARTIST for t in toks):
            return False
        sel.append(i)
        for t in toks:
            per_artist[t] = per_artist.get(t, 0) + 1
        seen.add(r.tkey)
        return True


    pool["is_xmas"] = pool.name.map(lambda t: bool(XMAS.search(str(t))))
    print("Weihnachtslieder im Pool (ausgeschlossen):", pool[pool.is_xmas][["name", "artists"]].values.tolist())
    pool["quota_pick"] = False
    _ord = pool.assign(adj=pool.score - PEN * (pool.voice == "male")).sort_values("adj", ascending=False)
    if EN_SHARE is not None:
        n_non = int(round((1 - EN_SHARE) * N))
        cap = max(2, int(np.ceil(MAX_LANG_SHARE * n_non)))
        per_lang = {}
        for i, r in _ord[(_ord.lang != "en") & (_ord.lang != "unknown")].iterrows():
            if len(sel) >= n_non:
                break
            if per_lang.get(r.lang, 0) >= cap:
                continue
            if take(i, r):
                pool.loc[i, "quota_pick"] = True
                per_lang[r.lang] = per_lang.get(r.lang, 0) + 1
        print(f"Sprachquote: {len(sel)} nicht-englische Songs (Ziel {n_non}), Sprachen: {per_lang}")
        # Rest: nur englisch/unklar, damit der Zielanteil nicht ueberschritten wird
        for i, r in _ord[(_ord.lang == "en") | (_ord.lang == "unknown")].iterrows():
            if len(sel) >= N:
                break
            take(i, r)
    for i, r in _ord.iterrows():
        if len(sel) >= N:
            break
        if i not in sel:
            take(i, r)
    if len(sel) < N:  # nicht genug Songs ueber der Schwelle: mit den besten darunter auffuellen und melden
        print(f"ACHTUNG: nur {len(sel)} Songs mit Passungsmass >= {TASTE_MIN}; Rest wird mit den besten darunter aufgefuellt")
        for i, r in _ord.iterrows():
            if len(sel) >= N:
                break
            if i not in sel:
                take(i, r, strict=False)
    return sel


def _male_share(ids):
    v = pool.loc[ids, "voice"]
    m, f = int((v == "male").sum()), int((v == "female").sum())
    return (m / (m + f) if m + f else float("nan")), m, f


if MALE_CAP is not None:
    for PEN in np.arange(0.0, 3.01, 0.25):
        _t = _select(N)
        _sh = _male_share(_t)
        if _sh[0] <= MALE_CAP or np.isnan(_sh[0]):
            break
    print(f"Stimm-Abzug fuer Maenner: {PEN:.2f} -> in der Auswahl {_sh[1]} Maenner, {_sh[2]} Frauen (Anteil {_sh[0]:.0%}, Ziel hoechstens {MALE_CAP:.0%})")
_cand_sel = _select(200)  # Kandidatenliste fuer die Recherche (gleiche Regeln und Quoten, 200 statt 75)
pool["candidate"] = pool.index.isin(_cand_sel)
sel = _select(N)
pool["selected"] = pool.index.isin(sel)

import os

os.makedirs("data/rankings", exist_ok=True)
keep_cols = ["rank", "selected", "quota_pick", "lang", "score", "breadth", "pos", "artist_hits", "track_id", "name", "artists",
             "n_playlists", "best_pos", "is_chart", "playlists", "feat", "stylefit", "voice"]
pool[keep_cols].to_csv(f"data/rankings/ranking_{DATE}{SUFFIX}.csv", index=False)
_c = pool[pool.candidate].reset_index(drop=True)
_c.insert(0, "nr", np.arange(1, len(_c) + 1))
_c[["nr", "track_id", "name", "artists", "score", "feat", "stylefit", "n_playlists", "best_pos", "lang", "voice", "voice_src", "style_tags", "style_flag", "genres_sp", "lf_tags", "playlists", "selected"]].to_csv(
    f"data/rankings/candidates_{DATE}{SUFFIX}.csv", index=False)
print(f"Kandidatenliste: {len(_c)} Songs, davon {int(_c.selected.sum())} in den Top 75")
top = pool[pool.selected].reset_index(drop=True)
top.insert(0, "nr", np.arange(1, len(top) + 1))
top[["nr", "track_id", "name", "artists", "score", "n_playlists", "best_pos", "lang"]].to_csv(
    f"data/rankings/playlist_{DATE}{SUFFIX}.csv", index=False
)
print(f"{len(top)} Songs ausgewaehlt, {top.artists.nunique()} Kuenstler")
print(top[["nr", "name", "artists", "n_playlists", "best_pos", "feat"]].to_string(max_colwidth=38))
