# Tills Geschmack - Massstab fuer die Song-Recherche

Quelle: seine Charts (data/charts/), seine Stil-Tags (data/analysis/mb_tags.csv) und sein Feedback (data/feedback.csv, Spalte reason).
Nur das zaehlt, was Till selbst gesagt oder in seinen Charts gezeigt hat. Nichts dazuerfinden.

## Das passt (haeufig in seinen Charts)
Pop, Electropop, Dance-Pop, Alternative Pop, Indie Pop, Synth-Pop, Hyperpop, Girl Groups, K-Pop, Singer-Songwriter,
franzoesischer Pop/Chanson-Pop, Folk-Pop, einzelne Indie-/Post-Punk-/Rock-Acts (z.B. Fontaines D.C.), Country-Pop vereinzelt,
Club/House nur mit Pop-Charakter. Viele lokale Songs in den Landessprachen der 14 NMF-Laender.
Weibliche Stimmen oefter als maennliche (Till: "maennliche Stimmen seltener als weibliche").

## Das passt nicht (von Till ausgeschlossen)
- Hardrock, Rock im Riff-/Stadionstil (Royal Blood, Kings of Leon, Cage The Elephant)
- Metal in jeder Form: Metalcore, Grindcore, Deathcore, Heavy/Thrash/Death/Black Metal, Sludge, Doom, Screamo, Post-Hardcore
- Grime und britischer Rap (Skepta, Jack Harlow), Trap, generell Rap/Hip-Hop ohne Pop-Charakter
- Soul / Neo-Soul / Retro-Soul (Christine and the Queens "zu soulig", Neema Nekesa)
- "Generic Dance" / EDM / Afro-House-Massenware (Shimza, ATRIP)
- Kinderlied-haft (Razem von Małe TGD)
- Belanglose, langweilige Songs (Petra - I Am God, Becky and the Birds)
- Stimme, die ihm nicht liegt (Onderweg), Latin-Pop-Massenware (COMO FUERA)
- Weihnachtslieder
Details und Begruendungen: data/feedback.csv (signal=dislike).

## Wie urteilen
- verdict `no`: nur wenn die Recherche klar zeigt, dass Stil/Charakter in die Ausschlussliste faellt oder dem Geschmack eindeutig widerspricht.
- verdict `ok`: Stil passt erkennbar zu den Charts.
- verdict `unsure`: Quellenlage duenn oder gemischt. Im Zweifel `unsure`, nie `no`.
- Pro Song 1 Zeile: kurze Stilbeschreibung (max. 12 Woerter), Grund (max. 20 Woerter), Quelle(n) als URL.
- Bewerte den Song/Kuenstler, nicht Popularitaet. Keine Meinung zu Personen, nur Musik.
