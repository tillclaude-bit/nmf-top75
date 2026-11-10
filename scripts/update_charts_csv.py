#!/usr/bin/env python3
"""Neue Top-100-Chartwoche aus einem Exportify-Export (Playlistreihenfolge = Platz) einlesen und data/charts/* aktualisieren.
Aufruf: python update_charts_csv.py EXPORT.csv CHART_DATUM
Zuordnung: erst ueber die Spotify-Track-ID (song_features), dann Kuenstler/Titel (wie update_charts.py).
Vorplatz = Platz in der letzten Chartwoche, Wochenzahl = Vorwoche + 1; sonst NEU (Wiedereinstieg: Wochen zaehlen neu bei 1)."""
import difflib, re, sys
import pandas as pd

path, cdate = sys.argv[1:3]
rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")


def norm(s):
    s = re.sub(r"[\(\[].*?[\)\]]", " ", str(s)); s = re.split(r"\s-\s", s)[0]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def tok(a):
    t = [norm(x) for x in re.split(r",|;|\xa0|&| feat\.| ft\.| x ", str(a)) if norm(x)]
    return t[0] if t else ""


ex = rd(path)
ex["tid"] = ex["Track URI"].str.split(":").str[-1]
assert len(ex) == 100, f"keine vollstaendige Top 100 ({len(ex)} Zeilen)"
ch = pd.DataFrame({"pos": range(1, 101), "tid": ex.tid, "artist": ex["Artist Name(s)"].str.replace(";", "; ").str.replace("⁠", ""),
                   "title": ex["Track Name"].str.replace("⁠", "")})
ch["key"] = ch.artist.map(tok) + "|" + ch.title.map(norm)
songs = rd("data/charts/songs.csv"); cr = rd("data/charts/chart_rows.csv"); sf = rd("data/charts/song_features.csv")
assert cdate not in set(cr.chart_date), "Chartwoche schon vorhanden"
songs["key"] = songs.artist.map(tok) + "|" + songs.title.map(norm)
kmap = dict(zip(songs.key, songs.song_id))
tmap = {t: s for t, s in zip(sf.spotify_track_id, sf.song_id) if isinstance(t, str) and not str(sf.loc[sf.song_id == s, "match_quality"].iloc[0]).startswith("falsch")}
byart = {}
for k, i in kmap.items():
    byart.setdefault(k.split("|")[0], []).append((k, i))
ids, how = [], {"id": 0, "key": 0, "fuzzy": 0, "neu": 0}
nxt = int(songs.song_id.str[1:].astype(int).max()) + 1
for r in ch.itertuples():
    sid = tmap.get(r.tid)
    if sid:
        how["id"] += 1
    else:
        sid = kmap.get(r.key)
        if sid:
            how["key"] += 1
        else:
            c = [(difflib.SequenceMatcher(None, r.key.split("|")[1], k.split("|")[1]).ratio(), i) for k, i in byart.get(r.key.split("|")[0], [])]
            if c and max(c)[0] >= 0.85:
                sid = max(c)[1]; how["fuzzy"] += 1
    if sid is None:
        sid = f"S{nxt:03d}"; nxt += 1; how["neu"] += 1
    ids.append(sid)
ch["song_id"] = ids
assert ch.song_id.is_unique, ch[ch.song_id.duplicated(keep=False)]
last = sorted(cr.chart_date.unique())[-1]
prev = cr[cr.chart_date == last].set_index("song_id")
ch["prev"] = ch.song_id.map(prev.pos)
ch["weeks"] = [int(prev.weeks_shown[s]) + 1 if s in prev.index else 1 for s in ch.song_id]
new_ids = [s for s in ids if s not in set(songs.song_id)]
reent = [(r.artist, r.title) for r in ch.itertuples() if r.song_id in set(songs.song_id) and pd.isna(r.prev)]
print(f"Zuordnung: {how}; Vorwoche {last}; Neueinsteiger {len(new_ids)}, Wiedereinstiege {len(reent)}: {reent}")

add = pd.DataFrame({"chart_date": cdate, "pos": ch.pos, "artist": ch.artist, "title": ch.title, "song_id": ch.song_id,
                    "prev_pos_shown": ch.prev, "prev_pos_checked": "", "weeks_shown": ch.weeks, "flag": ""})
cr = pd.concat([cr, add], ignore_index=True)

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

# song_features: Neueinsteiger direkt aus dem Export (enthaelt Spotify-Klangwerte)
exm = ex.set_index("tid")
fr = []
for r in ch[ch.song_id.isin(new_ids)].itertuples():
    e = exm.loc[r.tid]
    fr.append(dict(song_id=r.song_id, chart_artist=r.artist, chart_title=r.title, peak_pos=r.pos, weeks_in_10_charts=1,
                   spotify_track_id=r.tid, spotify_name=e["Track Name"], spotify_artist=e["Artist Name(s)"], isrc=e["ISRC"],
                   popularity=e["Popularity"], release_date=e["Release Date"], tempo=e["Tempo"], energy=e["Energy"],
                   danceability=e["Danceability"], valence=e["Valence"], acousticness=e["Acousticness"], loudness=e["Loudness"],
                   speechiness=e["Speechiness"], instrumentalness=e["Instrumentalness"], liveness=e["Liveness"], key=e["Key"],
                   mode=e["Mode"], match_quality="ok-export"))
for i, s in sf.iterrows():
    if s.song_id in pos.index:
        sf.loc[i, "weeks_in_10_charts"] = int(s.weeks_in_10_charts) + 1
        if pos.loc[s.song_id].pos < s.peak_pos:
            sf.loc[i, "peak_pos"] = int(pos.loc[s.song_id].pos)
sf = pd.concat([sf, pd.DataFrame(fr)], ignore_index=True)

cr2 = cr.merge(songs[["song_id"]], on="song_id")
cr2["artist_list"] = cr2.artist.str.split(";")
e2 = cr2.explode("artist_list"); e2["artist_list"] = e2.artist_list.str.strip().str.replace("\xa0", " ")
g = e2.groupby("artist_list").agg(songs_charted=("song_id", "nunique"), chart_weeks_total=("song_id", "size"),
                                  best_pos=("pos", "min"), weeks_in_top10=("pos", lambda x: int((x <= 10).sum()))).reset_index()
g = g.rename(columns={"artist_list": "artist"}).sort_values(["songs_charted", "chart_weeks_total"], ascending=False)
bom = dict(encoding="utf-8-sig", index=False)
cr.to_csv("data/charts/chart_rows.csv", **bom); songs.to_csv("data/charts/songs.csv", **bom)
sf.to_csv("data/charts/song_features.csv", **bom); g.to_csv("data/charts/artists.csv", **bom)
print("fertig:", len(cr), "Zeilen,", len(songs), "Songs,", len(g), "Kuenstler")
print(ch[ch.song_id.isin(new_ids)][["pos", "artist", "title"]].to_string(index=False))
