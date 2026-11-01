#!/usr/bin/env python3
"""Erkennt die Gesangssprache aus den 30-s-Vorschauen (Whisper, nur Spracherkennung, kein Transkript).
Gleiche Methode fuer Pool- und Chartsongs.
Aufruf: python language_id.py OUT.csv POOL.csv   (Chartsongs aus data/charts/song_features.csv werden mitgenommen)"""
import csv, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from preview_features import http_get, preview_url  # noqa
from faster_whisper import WhisperModel, decode_audio

out, pool = sys.argv[1], sys.argv[2]
items = {}
for r in csv.DictReader(open(pool, encoding="utf-8-sig")):
    items[r["track_id"]] = ("pool", r.get("preview_url") or None)
for r in csv.DictReader(open("data/charts/song_features.csv", encoding="utf-8-sig")):
    if r.get("match_quality") == "ok" and r.get("spotify_track_id"):
        items.setdefault(r["spotify_track_id"], ("chart", None))
# Chartsongs, deren Sprache schon aus frueheren Laeufen bekannt ist, nicht erneut berechnen
import glob
done = set()
for f in glob.glob("data/analysis/language_*.csv"):
    if os.path.abspath(f) != os.path.abspath(out):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            if r.get("lang"):
                done.add(r["track_id"])
items = {k: v for k, v in items.items() if v[0] == "pool" or k not in done}
print(f"{len(items)} Tracks", flush=True)


def fetch(kv):
    tid, (src, url) = kv
    try:
        url = url or preview_url(tid)
        if not url:
            return tid, src, None, "keine Preview-URL"
        with tempfile.TemporaryDirectory() as t:
            mp3, wav = os.path.join(t, "p.mp3"), os.path.join(t, "p.wav")
            open(mp3, "wb").write(http_get(url))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ac", "1", "-ar", "16000", wav], check=True)
            return tid, src, decode_audio(wav, sampling_rate=16000), ""
    except Exception as e:
        return tid, src, None, repr(e)[:100]


model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
rows = []
with ThreadPoolExecutor(max_workers=4) as ex:
    for tid, src, audio, err in ex.map(fetch, items.items()):
        row = {"track_id": tid, "source": src, "lang": "", "prob": "", "lang2": "", "prob2": "", "error": err}
        if audio is not None:
            try:
                _, info = model.transcribe(audio, language=None, beam_size=1, without_timestamps=True)
                probs = sorted(info.all_language_probs or [(info.language, info.language_probability)], key=lambda x: -x[1])[:2]
                row.update(lang=probs[0][0], prob=round(probs[0][1], 3),
                           lang2=probs[1][0] if len(probs) > 1 else "", prob2=round(probs[1][1], 3) if len(probs) > 1 else "")
            except Exception as e:
                row["error"] = repr(e)[:100]
        rows.append(row)
        if len(rows) % 100 == 0:
            print(len(rows), flush=True)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
ok = sum(1 for r in rows if r["lang"])
print(f"Sprache erkannt: {ok} von {len(rows)}")
