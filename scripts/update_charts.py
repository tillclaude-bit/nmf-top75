#!/usr/bin/env python3
"""Liest eine neue Top-100-Chartwoche (Word-Datei) und aktualisiert data/charts/*.
Aufruf: python update_charts.py CHART.docx CHART_DATUM POOL_DATUM
  CHART_DATUM  Datum der neuen Chartwoche (JJJJ-MM-TT)
  POOL_DATUM   NMF-Woche, aus der neue Songs ihre Spotify-ID und Klang-Eigenschaften bekommen
Zeilenformat: '01 Artist; Artist – Titel (Vorplatz|NEW) 5w'"""
import difflib, re, sys
import docx
import pandas as pd

docx_path, cdate, pdate = sys.argv[1:4]


def norm(s):
    s = re.sub(r"[\(\[].*?[\)\]]", " ", str(s)); s = re.split(r"\s-\s", s)[0]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def tok(a):
    t = [norm(x) for x in re.split(r",|;|\xa0|&| feat\.| ft\.| x ", str(a)) if norm(x)]
    return t[0] if t else ""


rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")
doc = docx.Document(docx_path)
text = "\n".join(p.text for p in doc.paragraphs)
rows = []
for l in [l for l in text.split("\n") if l.strip()]:
    m = re.match(r"^(\d+)\s+(.*?)\s+–\s+(.*?)\s+\((NEW|\d+)\)(?:\s+(\d+)w)?\s*$", l)
    if not m:
        sys.exit(f"Zeile nicht lesbar: {l!r}")
    pos, a, t, prev, w = m.groups()
    rows.append(dict(pos=int(pos), artist=a.strip(), title=t.strip(), prev=None if prev == "NEW" else int(prev),
                     weeks=1 if prev == "NEW" else int(w)))
ch = pd.DataFrame(rows)
assert len(ch) == 100 and sorted(ch.pos) == list(range(1, 101)), "keine vollstaendige Top 100"
ch["key"] = ch.artist.map(tok) + "|" + ch.title.map(norm)

songs = rd("data/charts/songs.csv"); cr = rd("data/charts/chart_rows.csv"); sf = rd("data/charts/song_features.csv")
assert cdate not in set(cr.chart_date), "Chartwoche schon vorhanden"
songs["key"] = songs.artist.map(tok) + "|" + songs.title.map(norm)
kmap = dict(zip(songs.key, songs.song_id))
byart = {}
for k, i in kmap.items():
    byart.setdefault(k.split("|")[0], []).append((k, i))
ids, fuzzy = [], []
nxt = int(songs.song_id.str[1:].astype(int).max()) + 1
for r in ch.itertuples():
    sid = kmap.get(r.key)
    if sid is None and r.prev is not None:  # Schreibweise abweichend: gleicher Kuenstler, aehnlicher Titel
        c = [(difflib.SequenceMatcher(None, r.key.split("|")[1], k.split("|")[1]).ratio(), i)
             for k, i in byart.get(r.key.split("|")[0], [])]
        if c and max(c)[0] >= 0.8:
            sid = max(c)[1]; fuzzy.append((r.artist, r.title, sid))
    if sid is None:
        sid = f"S{nxt:03d}"; nxt += 1
    ids.append(sid)
ch["song_id"] = ids
print("unscharf zugeordnet:", fuzzy)
new_ids = [s for s in ids if s not in set(songs.song_id)]
print(f"{len(new_ids)} Neueinsteiger")

# chart_rows
add = pd.DataFrame({"chart_date": cdate, "pos": ch.pos, "artist": ch.artist, "title": ch.title, "song_id": ch.song_id,
                    "prev_pos_shown": ch.prev, "prev_pos_checked": "", "weeks_shown": ch.weeks, "flag": ""})
cr = pd.concat([cr, add], ignore_index=True)

# songs
songs = songs.drop(columns="key")
pos = ch.set_index("song_id")
for i, s in songs.iterrows():
    if s.song_id in pos.index:
        p = pos.loc[s.song_id]
        songs.loc[i, ["last_seen", "last_pos", "weeks_total_shown_last", "in_latest_chart"]] = [cdate, int(p.pos), int(p.weeks), "ja"]
        songs.loc[i, "weeks_in_10_charts"] = int(s.weeks_in_10_charts) + 1
        if p.pos < s.peak_pos:
            songs.loc[i, ["peak_pos", "peak_first_date"]] = [int(p.pos), cdate]
    else:
        songs.loc[i, "in_latest_chart"] = "nein"
newrows = [dict(song_id=r.song_id, artist=r.artist, title=r.title, first_seen=cdate, entry_pos=r.pos, entered_in_window="ja",
                peak_pos=r.pos, peak_first_date=cdate, weeks_in_10_charts=1, last_seen=cdate, last_pos=r.pos,
                weeks_total_shown_last=r.weeks, in_latest_chart="ja") for r in ch.itertuples() if r.song_id in new_ids]
songs = pd.concat([songs, pd.DataFrame(newrows)], ignore_index=True)

# song_features / Klang-Eigenschaften der Neueinsteiger aus dem NMF-Pool
pool = rd(f"data/pools/pool_tracks_{pdate}.csv"); pdesc = rd(f"data/pools/pool_descriptors_{pdate}.csv")
pool["key"] = pool.artists.map(tok) + "|" + pool.name.map(norm)
pm = pool.drop_duplicates("key").set_index("key")
fr, dr, miss = [], [], []
for r in ch[ch.song_id.isin(new_ids)].itertuples():
    if r.key in pm.index:
        t = pm.loc[r.key]
        fr.append(dict(song_id=r.song_id, chart_artist=r.artist, chart_title=r.title, peak_pos=r.pos, weeks_in_10_charts=1,
                       spotify_track_id=t.track_id, spotify_name=t["name"], spotify_artist=t.artists, match_quality="ok-pool"))
        d = pdesc[pdesc.track_id == t.track_id]
        if len(d):
            dr.append(d.iloc[0].to_dict() | {"source": "chart"})
    else:
        miss.append((r.artist, r.title))
        fr.append(dict(song_id=r.song_id, chart_artist=r.artist, chart_title=r.title, peak_pos=r.pos, weeks_in_10_charts=1,
                       match_quality="nicht im NMF-Pool"))
print("nicht im NMF-Pool (ohne Klang-Eigenschaften):", miss)
for i, s in sf.iterrows():
    if s.song_id in pos.index:
        sf.loc[i, "weeks_in_10_charts"] = int(s.weeks_in_10_charts) + 1
        if pos.loc[s.song_id].pos < s.peak_pos:
            sf.loc[i, "peak_pos"] = int(pos.loc[s.song_id].pos)
sf = pd.concat([sf, pd.DataFrame(fr)], ignore_index=True)
if dr:
    pd.DataFrame(dr).to_csv(f"data/analysis/preview_descriptors_{cdate}.csv", index=False)

# artists
cr2 = cr.merge(songs[["song_id"]], on="song_id")
cr2["artist_list"] = cr2.artist.str.split(";")
ex = cr2.explode("artist_list"); ex["artist_list"] = ex.artist_list.str.strip().str.replace("\xa0", " ")
g = ex.groupby("artist_list").agg(songs_charted=("song_id", "nunique"), chart_weeks_total=("song_id", "size"),
                                  best_pos=("pos", "min"), weeks_in_top10=("pos", lambda x: int((x <= 10).sum()))).reset_index()
g = g.rename(columns={"artist_list": "artist"}).sort_values(["songs_charted", "chart_weeks_total"], ascending=False)
bom = lambda p: dict(encoding="utf-8-sig", index=False)
cr.to_csv("data/charts/chart_rows.csv", **bom(0)); songs.to_csv("data/charts/songs.csv", **bom(0))
sf.to_csv("data/charts/song_features.csv", **bom(0)); g.to_csv("data/charts/artists.csv", **bom(0))
print("fertig:", len(cr), "Zeilen,", len(songs), "Songs,", len(g), "Kuenstler")
