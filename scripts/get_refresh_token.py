#!/usr/bin/env python3
"""Einmaliges Hilfsskript: holt einen Spotify-Refresh-Token (PKCE, ohne Client Secret).

Ablauf:
  1. python get_refresh_token.py
  2. Client ID eingeben (steht im Spotify Developer Dashboard, ist kein Geheimnis)
  3. Der Browser öffnet die Spotify-Freigabe, dort mit "Zustimmen" bestätigen
  4. Das Skript fängt die Weiterleitung auf http://127.0.0.1:8888/callback ab
  5. Der ausgegebene Refresh-Token wird als GitHub-Secret SPOTIFY_REFRESH_TOKEN gespeichert

Der Token wird nur im Terminal angezeigt, nie in eine Datei geschrieben.
Voraussetzung: In der Spotify-App ist exakt diese Redirect-URI eingetragen:
  http://127.0.0.1:8888/callback
"""
import base64
import hashlib
import http.server
import json
import secrets
import threading
import urllib.parse
import urllib.request
import webbrowser

REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPES = "playlist-modify-public playlist-modify-private"


def main():
    client_id = input("Spotify Client ID: ").strip()
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    state = secrets.token_urlsafe(16)
    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "state": state,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
    )

    result = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result["code"] = query.get("code", [None])[0]
            result["state"] = query.get("state", [None])[0]
            result["error"] = query.get("error", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                "Fertig - du kannst dieses Fenster schliessen und zum Terminal zurueckkehren.".encode()
            )
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 8888), Handler)
    print("Öffne Browser zur Spotify-Freigabe ...")
    print("Falls sich nichts öffnet, diese URL im Browser aufrufen:\n" + auth_url + "\n")
    webbrowser.open(auth_url)
    server.serve_forever()

    if result.get("error") or not result.get("code"):
        raise SystemExit(f"Freigabe fehlgeschlagen: {result.get('error')}")
    if result.get("state") != state:
        raise SystemExit("State stimmt nicht überein - Abbruch.")

    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
        }
    ).encode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req) as resp:
        tokens = json.load(resp)

    print("Erteilte Rechte:", tokens.get("scope"))
    print("\nREFRESH TOKEN (als GitHub-Secret SPOTIFY_REFRESH_TOKEN speichern):\n")
    print(tokens["refresh_token"])


if __name__ == "__main__":
    main()
