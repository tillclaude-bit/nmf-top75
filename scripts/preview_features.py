#!/usr/bin/env python3
"""Berechnet eigene Audio-Deskriptoren aus den 30-Sekunden-Spotify-Vorschauen.

Zweck: Spotifys Audio-Features sind fuer unsere App gesperrt. Dieses Skript holt pro
Track-ID die Vorschau-URL (Embed-Seite), laedt die MP3 und berechnet Deskriptoren
(Tempo, Lautheit, Spektrum, Rhythmus, Tonart, MFCC ...). Mit den Songs, zu denen wir
echte Spotify-Features haben, kalibrieren wir spaeter ein Modell, das daraus
Spotify-aehnliche Werte schaetzt.

Aufruf: python preview_features.py OUT.csv [--limit N]
Eingabe: IDs aus data/pools/pool_sample_features_*.csv und data/charts/song_features.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def preview_url(track_id):
    import time, random
    html = ""
    for i in range(4):  # Spotify drosselt die Embed-Seite bei vielen Abfragen: mit Pause wiederholen
        try:
            html = http_get(f"https://open.spotify.com/embed/track/{track_id}").decode("utf-8", "replace")
            if "audioPreview" in html or "mp3-preview" in html:
                break
        except Exception:
            pass
        time.sleep(2 + 3 * i + random.random())
    m = re.search(r'"audioPreview"\s*:\s*\{\s*"url"\s*:\s*"(https:[^"]+)"', html)
    if not m:
        m = re.search(r"(https://p\.scdn\.co/mp3-preview/[0-9a-f]+[^\"'\\ ]*)", html)
    if not m:
        return None
    return m.group(1).replace("\\u002F", "/").replace("\\/", "/")


def _nrm(x):
    import unicodedata
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"\(.*?\)|\[.*?\]| - .*$", " ", x)  # Zusaetze wie (feat. X) oder "- Remastered" weg
    return re.sub(r"[^a-z0-9]+", " ", x).strip()


def deezer_preview(artist, title):
    """Fallback fuer Songs ohne Spotify-Vorschau: 30-s-Vorschau von Deezer (Suche nach Kuenstler + Titel, strenger Namensabgleich)."""
    import urllib.parse, time
    main_art = re.split(r",|;| feat\.| ft\.| x | & ", str(artist))[0].strip()
    ta, tt = _nrm(main_art), _nrm(title)
    if not ta or not tt:
        return None
    q = urllib.parse.quote(f'artist:"{main_art}" track:"{re.sub(r"[(].*", "", str(title)).strip()}"')
    for i in range(3):
        try:
            time.sleep(0.3)
            data = json.loads(http_get(f"https://api.deezer.com/search?q={q}&limit=8"))
            for it in data.get("data", []):
                if it.get("preview") and _nrm(it["artist"]["name"]) == ta and (_nrm(it["title"]) == tt or tt in _nrm(it["title"]) or _nrm(it["title"]) in tt):
                    return it["preview"]
            return None
        except Exception:
            time.sleep(2 + 2 * i)
    return None


# Krumhansl-Kessler-Profile fuer Tonart und Modus
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_mode(chroma_mean):
    best = (-2.0, 0, 1)
    for k in range(12):
        for mode, prof in ((1, MAJOR), (0, MINOR)):
            c = np.corrcoef(np.roll(prof, k), chroma_mean)[0, 1]
            if c > best[0]:
                best = (c, k, mode)
    return best[1], best[2], best[0]


def describe(wav_path):
    import librosa

    y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=30)
    d = {"dur": len(y) / sr}
    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    d["rms_db_mean"] = float(np.mean(rms_db))
    d["rms_db_p95"] = float(np.percentile(rms_db, 95))
    d["dyn_range_db"] = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    d["peak_db"] = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    try:
        import pyloudnorm as pyln

        d["lufs"] = float(pyln.Meter(sr).integrated_loudness(y))
    except Exception:
        d["lufs"] = float("nan")
    S = np.abs(librosa.stft(y))
    d["centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    d["bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    d["rolloff85"] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    d["flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    d["contrast"] = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    d["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    d["onset_mean"] = float(np.mean(oenv))
    d["onset_std"] = float(np.std(oenv))
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    d["tempo"] = float(np.atleast_1d(tempo)[0])
    d["n_beats"] = int(len(beats))
    bt = librosa.frames_to_time(beats, sr=sr)
    ibi = np.diff(bt)
    d["beat_cv"] = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 2 else float("nan")
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr).mean(axis=1)[1:]
    d["pulse_clarity"] = float(np.max(tg) / (np.mean(tg) + 1e-9))
    yh, yp = librosa.effects.hpss(y)
    d["hp_ratio_db"] = float(
        20 * np.log10((np.sqrt(np.mean(yh**2)) + 1e-9) / (np.sqrt(np.mean(yp**2)) + 1e-9))
    )
    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr).mean(axis=1)
    key, mode, strength = key_mode(chroma)
    d["key"], d["mode"], d["key_strength"] = key, mode, float(strength)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        d[f"mfcc{i+1}_mean"] = float(mf[i].mean())
        d[f"mfcc{i+1}_std"] = float(mf[i].std())
    return d


def process(tid, url=None, meta=None):
    row = {"track_id": tid, "preview_ok": 0, "error": "", "preview_src": "spotify"}
    try:
        url = url or preview_url(tid)
        if not url and meta:
            url = deezer_preview(*meta)
            row["preview_src"] = "deezer"
        if not url:
            row["error"] = "keine Preview-URL"
            row["preview_src"] = ""
            return row
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = os.path.join(tmp, "p.mp3")
            wav = os.path.join(tmp, "p.wav")
            with open(mp3, "wb") as f:
                f.write(http_get(url))
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "22050", wav],
                check=True,
            )
            row.update(describe(wav))
            row["preview_ok"] = 1
    except Exception as e:
        row["error"] = repr(e)[:160]
    return row


def load_ids():
    ids = {}
    for p in sorted(glob.glob("data/pools/pool_sample_features_*.csv")):
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("track_id") and r.get("energy"):
                    ids[r["track_id"]] = "pool_sample"
    with open("data/charts/song_features.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("match_quality") == "ok" and r.get("spotify_track_id"):
                ids.setdefault(r["spotify_track_id"], "chart")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", help="pool_tracks_*.csv (track_id, preview_url) statt Kalibriersatz")
    ap.add_argument("--only-missing", action="store_true", help="nur Pool-Songs ohne Spotify-Vorschau (Test des Deezer-Fallbacks)")
    a = ap.parse_args()
    urls, metas = {}, {}
    if a.pool:
        ids = {}
        with open(a.pool, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                ids[r["track_id"]] = "pool"
                urls[r["track_id"]] = r.get("preview_url") or None
                metas[r["track_id"]] = (r.get("artists", ""), r.get("name", ""))
    else:
        ids = load_ids()
    items = list(ids.items())
    if a.only_missing:
        items = [kv for kv in items if not urls.get(kv[0])]
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} Tracks", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(lambda kv: process(kv[0], urls.get(kv[0]), metas.get(kv[0])), items))
    for r, (_, src) in zip(rows, items):
        r["source"] = src
    cols = ["track_id", "source", "preview_ok", "error"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    ok = sum(r["preview_ok"] for r in rows)
    print(f"Vorschau ok: {ok} von {len(rows)} (davon Deezer-Fallback: {sum(1 for r in rows if r['preview_ok'] and r.get('preview_src') == 'deezer')})")
    errs = {}
    for r in rows:
        if r["error"]:
            errs[r["error"][:60]] = errs.get(r["error"][:60], 0) + 1
    for e, n in sorted(errs.items(), key=lambda x: -x[1])[:5]:
        print(f"  Fehler x{n}: {e}")


if __name__ == "__main__":
    sys.exit(main())
