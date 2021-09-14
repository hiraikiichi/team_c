import os
from os.path import join, dirname
from dotenv import load_dotenv
import sys
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from flask import Flask, render_template, request

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)

TOKEN = os.environ.get("TEST")
# TOKEN = 334
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
CLIENT_CREDENTIALS_MANAGER = spotipy.oauth2.SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=CLIENT_CREDENTIALS_MANAGER)

@app.route('/')
def hello_world():
    ver = sys.version
    # target = os.environ.get('TARGET', 'World')
    return render_template('index.html')

# "/" →　"〇〇.html"の結果表示のとこへ変更する
@app.route("/",methods=["POST"])
def show():
    title = request.form["title"]
    return "ようこそ、" + title + "さん"

@app.route('/about')
def aboutPage():
    return render_template('about.html')

def get_songs_from_playlist(playlist_id: str):
    songs = []
    result = spotify.playlist(playlist_id)
    for item in result['tracks']['items']:
        song_name = item['track']['name']
        artist = item['track']['artists'][0]['name']
        ref = item['track']['href']
        # preview_url = item['track']['preview_url']
        songs.append([song_name, artist, ref])
    return songs

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8888)))