# NMF Top 75

Wöchentliche Top-75-Playlist aus den Spotify-New-Music-Friday-Playlists von 14 Ländern
(UK, DE, AU/NZ, FR, ES, IT, GR, DK, NO, SE, FI, KR, PL, NL).

## Struktur

- `data/charts/`: eigene Charts (10 Wochen, 26.8. bis 28.10.2026) als CSV
  - `chart_rows.csv`: alle Chartzeilen (Platz, Künstler, Titel, Vorwoche, Wochen, NEW/RE)
  - `songs.csv`: eindeutige Songs mit Peak, Wochen in den Charts, Ersteintritt
  - `artists.csv`: Künstler mit Anzahl Songs, Chartwochen, bester Position
- `data/pools/`: wöchentlicher Pool aller 14 Playlists (folgt)
- `data/rankings/`: wöchentlich das vollständige Ranking aller neuen Songs (folgt)
- `data/feedback.csv`: Songs, die nicht gefallen

## Hinweise zu den Chart-Daten

- `prev_pos_shown` und `weeks_shown` sind so übernommen, wie sie im Chart-Bild stehen.
  `prev_pos_checked` ist die per Abgleich mit dem Vorchart berechnete Vorwoche.
- Im Chart vom 2.9. stehen für KiiiKiii, FLO und Shye falsche Vorwochen (45, 46, 65 statt 42, 44, 38),
  am 7.10. hat Francesca Michielin 3 statt 2 Wochen. Das sind Fehler in der Vorlage.
- Peak und Wochen gelten nur innerhalb dieser 10 Charts.

## Zugangsdaten

Apify-Token und Spotify-Zugang gehören nur als GitHub-Secrets ins Repo, nie in Dateien.
