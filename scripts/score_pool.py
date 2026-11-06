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
W = {"taste": 1.0, "breadth": 0.5, "pos": 0.25, "artist": 0.5, "stylefit": 0.5}  # style: Stil-Tags der Kuenstler (MusicBrainz) gegen die Stile der Chart-Kuenstler; ohne Tags neutral (0)  # breadth bleibt drin (Till, 2026-10-31); Vorwochen-Backtest: Breite sagt Chart-Einstiege vorher
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
pdesc = rd(f"data/pools/pool_descriptors_{DATE}.csv")
import glob as _g
cdesc = pd.concat([rd(f) for f in sorted(_g.glob("data/analysis/preview_descriptors_*.csv"))], ignore_index=True)
cdesc = cdesc[cdesc.source == "chart"].drop_duplicates("track_id", keep="last")
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
likes = (set(fb[fb.signal == "like"].track_id.astype(str)) if len(fb) else set()) - set(cd.spotify_track_id)  # Chartsongs sind schon Positive
is_like = pool.track_id.isin(likes).values
# Gefallene Songs (auch aus frueheren Wochen) zaehlen als zusaetzliche Positive (Gewicht 2), nie als Negative
neutral = set(fb[fb.signal == "neutral"].track_id.astype(str)) if len(fb) else set()
weak = set(fb[fb.signal == "weak_dislike"].track_id.astype(str)) if len(fb) else set()
neg = neg & ~is_like & ~pool.track_id.isin(neutral).values  # neutral: weder Positiv noch Negativ
import glob
hist = pd.concat([rd(f) for f in sorted(glob.glob("data/pools/pool_descriptors_*.csv"))], ignore_index=True)
hist = hist[hist.track_id.isin(likes) & (hist.preview_ok == 1)].drop_duplicates("track_id")
Xl = hist[DESC].reset_index(drop=True)
Xl = Xl.fillna(pd.concat([Xc, Xp]).median())
X = pd.concat([Xc, Xl, Xp[neg]], ignore_index=True)
y = np.r_[np.ones(len(Xc) + len(Xl)), np.zeros(neg.sum())]
w = np.r_[cd.w.values, np.full(len(Xl), 2.0), np.where(pool.track_id[neg].isin(weak).values, 2.0, 1.0)]
off = len(Xc) + len(Xl)
print(f"Zusaetzliche Positive aus deinem Feedback: {len(Xl)}")


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

# Breitenbonus nur, wenn der Klang passt (Till, 2026-11-06): bei unterdurchschnittlichem Passungsmass wird ein
# positiver Breitenbeitrag gedaempft (Faktor 1 + z_taste/2, mindestens 0); Backtest 30.10.: etwas mehr Likes, weniger Dislikes in den Top 75
_zs = {k: np.nan_to_num(z(pool[k]), nan=0.0) for k in W}  # fehlende Stil-Passung (keine Tags) zaehlt neutral
_g = np.clip(1 + _zs["taste"] / 2.0, 0, 1)
_zs["breadth"] = np.where((_zs["breadth"] > 0) & (_zs["taste"] < 0), _zs["breadth"] * _g, _zs["breadth"])
pool["score"] = sum(W[k] * _zs[k] for k in W)
pool.loc[pool.preview_ok != 1, "score"] -= 0.5  # keine Deskriptoren: vorsichtiger Abschlag

# --- Feedback / Ausschluesse
bad = set(fb[fb.signal == "dislike"].track_id.astype(str)) if len(fb) else set()
pool = pool[~pool.track_id.isin(bad)].reset_index(drop=True)

pool["style_excl"] = ""
if _os0.path.exists("data/style_rules.csv"):
    for _, _r in rd("data/style_rules.csv").iterrows():
        if _r.effect != "exclude":
            continue
        _t = _r.term.lower()
        _contains = ("match" in _r.index) and str(_r.match) == "contains"
        def _m(tag):
            tag = tag.split(":")[0].strip().lower().replace("-", " ")
            return (_t.replace("-", " ") in tag) if _contains else (tag == _t.replace("-", " "))
        hit = pool.style_tags.apply(lambda t: any(_m(x) for x in [y for y in t.split(";") if y][:2]))  # nur die 2 haeufigsten Tags (schuetzt vor Namensverwechslungen und Randtags)
        pool.loc[hit & (pool.style_excl == ""), "style_excl"] = _r.term
    _ex = pool[pool.style_excl != ""]
    print(f"Stil-Ausschluesse: {len(_ex)}")
    _ex[["track_id", "name", "artists", "style_excl", "style_tags", "score"]].to_csv(f"data/rankings/style_excluded_{DATE}{SUFFIX}.csv", index=False)
    pool = pool[pool.style_excl == ""].reset_index(drop=True)
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

# --- Auswahl: keine Chart-Duplikate, max. 2 pro Kuenstler, keine Doppelversionen
# Erst die besten nicht-englischen Songs bis zum Zielanteil (aus den aktuellen Charts), dann der Rest nach Score
sel, per_artist, seen = [], {}, set()


TASTE_MIN = 0.0  # Till, 2026-11-06: Songs mit Passungsmass unter dem Durchschnitt (Entscheidungsgrenze 0) kommen nicht in die 75
# (ohne Vorschau gibt es kein Passungsmass: diese Songs sind vom Filter ausgenommen)


def take(i, r, strict=True):
    if r.is_xmas:
        return False
    if strict and r.preview_ok == 1 and r.taste < TASTE_MIN:
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
if EN_SHARE is not None:
    n_non = int(round((1 - EN_SHARE) * N))
    cap = max(2, int(np.ceil(MAX_LANG_SHARE * n_non)))
    per_lang = {}
    for i, r in pool[(pool.lang != "en") & (pool.lang != "unknown")].iterrows():
        if len(sel) >= n_non:
            break
        if per_lang.get(r.lang, 0) >= cap:
            continue
        if take(i, r):
            pool.loc[i, "quota_pick"] = True
            per_lang[r.lang] = per_lang.get(r.lang, 0) + 1
    print(f"Sprachquote: {len(sel)} nicht-englische Songs (Ziel {n_non}), Sprachen: {per_lang}")
    # Rest: nur englisch/unklar, damit der Zielanteil nicht ueberschritten wird
    for i, r in pool[(pool.lang == "en") | (pool.lang == "unknown")].iterrows():
        if len(sel) >= N:
            break
        take(i, r)
for i, r in pool.iterrows():
    if len(sel) >= N:
        break
    if i not in sel:
        take(i, r)
if len(sel) < N:  # nicht genug Songs ueber der Schwelle: mit den besten darunter auffuellen und melden
    print(f"ACHTUNG: nur {len(sel)} Songs mit Passungsmass >= {TASTE_MIN}; Rest wird mit den besten darunter aufgefuellt")
    for i, r in pool.iterrows():
        if len(sel) >= N:
            break
        if i not in sel:
            take(i, r, strict=False)
pool["selected"] = pool.index.isin(sel)

import os

os.makedirs("data/rankings", exist_ok=True)
keep_cols = ["rank", "selected", "quota_pick", "lang", "score", "taste", "breadth", "pos", "artist_hits", "track_id", "name", "artists",
             "n_playlists", "best_pos", "preview_ok", "is_chart", "tempo", "rms_db_mean", "playlists"]
pool[keep_cols].to_csv(f"data/rankings/ranking_{DATE}{SUFFIX}.csv", index=False)
top = pool[pool.selected].reset_index(drop=True)
top.insert(0, "nr", np.arange(1, len(top) + 1))
top[["nr", "track_id", "name", "artists", "score", "n_playlists", "best_pos", "lang"]].to_csv(
    f"data/rankings/playlist_{DATE}{SUFFIX}.csv", index=False
)
print(f"{len(top)} Songs ausgewaehlt, {top.artists.nunique()} Kuenstler")
print(top[["nr", "name", "artists", "n_playlists", "best_pos", "taste"]].to_string(max_colwidth=38))
