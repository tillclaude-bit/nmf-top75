#!/usr/bin/env python3
"""Externe Pruefung der Sprache: Songtext aus LRCLIB (frei, ohne Schluessel) + Texterkennung.
Aufruf: python lyrics_lang.py OUT.csv POOL.csv"""
import csv, json, re, sys, time, urllib.parse, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0
out, pool = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(pool, encoding="utf-8-sig")))
UA = {"User-Agent": "nmf-top75/1.0 (personal project)"}


def first_artist(a):
    return re.split(r",|;|\xa0|&| feat\.| ft\.", a)[0].strip()


def get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + i)
        except Exception:
            time.sleep(1 + i)
    return None


def work(r):
    a, t = first_artist(r["artists"]), re.sub(r"\s*[\(\[].*?[\)\]]", "", r["name"]).split(" - ")[0].strip()
    q = urllib.parse.urlencode({"artist_name": a, "track_name": t, "duration": int(float(r["duration_ms"]) / 1000)})
    d = get("https://lrclib.net/api/get?" + q)
    if not d or d.get("instrumental"):
        d2 = get("https://lrclib.net/api/search?" + urllib.parse.urlencode({"artist_name": a, "track_name": t}))
        d = (d2 or [None])[0] if d2 else None
    txt = (d or {}).get("plainLyrics") or ""
    res = {"track_id": r["track_id"], "has_lyrics": int(len(txt) > 40), "lyr_lang": "", "lyr_prob": "", "lyr_lang2": "", "lyr_prob2": ""}
    if len(txt) > 40:
        try:
            ls = detect_langs(txt[:2000])
            res.update(lyr_lang=ls[0].lang, lyr_prob=round(ls[0].prob, 3))
            if len(ls) > 1:
                res.update(lyr_lang2=ls[1].lang, lyr_prob2=round(ls[1].prob, 3))
        except Exception:
            pass
    time.sleep(0.1)
    return res


with ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(work, rows))
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
print(f"{len(res)} Songs, Songtext gefunden: {sum(r['has_lyrics'] for r in res)}, Sprache erkannt: {sum(1 for r in res if r['lyr_lang'])}")
