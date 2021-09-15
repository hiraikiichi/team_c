import os
import requests
import random
from os.path import join, dirname
from dotenv import load_dotenv
import sys
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from flask import Flask, render_template, request, send_from_directory
import numpy as np
import math
from mlask import MLAsk

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)

TOKEN = os.environ.get("TEST")
# TOKEN = 334
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
CLIENT_CREDENTIALS_MANAGER = spotipy.oauth2.SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=CLIENT_CREDENTIALS_MANAGER)

APP_ID = os.environ.get("APP_ID") # applicationId(rakuten books api)

RECOMMEND_NUM = 4

RECOMMEND_PLAYLIST = {
    "6BGaNbk6J9JiPCjLAR3l3B": [-3, 4, -5, 3, 3, 0, 0, 2, 0, 5],
    "5sjNdkqhmvF0RLOUwSI3AW": [1, -1, -2, 2, 0, -1, 0, -2, -3, 0],
    "0g2CExISe9gl5tCK0fGsC7": [-3, 3, 0, 0, 2, 0, 0, -2, 3, 0],
    "7eRL4exJUcTsmiNGpXR31u": [-3, 2, -5, 0, 2, 0, -3, -2, -3, 0],
    "3rUgrOfEQAjNvwqPCFFNPB": [2, -2, 3, 0, 0, 2, 3, 3, 3, 0],
    "78qiR2mpH9x1bAe7Bg2wjk": [-3, 3, -3, 0, 5, -3, -3, 1, 3, 0],
    "5FUKrdhKliq1eSyQP8ioZ0": [-3, 3, -1, 0, 3, 0, 0, 2, 5, 0],
    "5dQV25DPP9rfhUkOOh5VcD": [-3, 3, -3, 0, 4, -3, -3, 2, 3, 0],
    "0s4S7bzYygOGXEzF1a8c4f": [-3, 0, -3, 0, 2, 0, 0, -2, -2, 0],
    "0axQfChvnswvoQUpBTSomE": [-1, 3, -3, 0, 1, 0, 0, 1, 3, 0],
    "3mBDzf6mlMOdtaYUyv2pSP": [-3, 3, -2, 0, 3, 0, -2, 3, 5, 0],
    "2kvS4vbHe5X9VSpyrvfPmg": [2, 1, 1, 0, 2, 1, -2, -2, -3, 0],
    "3UjF48ermq8mvO5m5PGjg4?si=135a86d81fc449f9&nd=1": [-5, 4, 2, 3, 0, -4, 3, 0, 5, 0],
    "4jGnTsYCkQh5OiVUpocHOC?si=8170c15fa6d044c6": [-3, 3, -3, 4, 3, 0, -3, 2, 2, 0],
    "5fuWZQ1o4ViDHoPP1IZ0QJ": [-3, 2, -2, 0, 3, 2, -2, -2, 4, 0],
    "37bzS3YW45JAEwcCgQtxGj": [0, 2, 0, 0, 3, 0, 0, 3, 4, 0],
    "7HmSjcMKUgmKYiVatRTvDM?si=2b744968547b4365": [0, 4, -5, 0, 1, -2, 0, 2, 4, 0],
    "3dMbFdufD0w1EtPnKEDDEg?si=0cf538a4ff6b4526": [-4, 4, 0, 4, 3, 0, 1, 1, 2, 0]
}

for k,v in RECOMMEND_PLAYLIST.items():
  n = list(map(lambda x: x/10, RECOMMEND_PLAYLIST[k]))
  RECOMMEND_PLAYLIST[k] =  n 

#分析(analyzer,feature)
analyzer = MLAsk()
features = ["iya","yorokobi","kowa","yasu","suki","aware","ikari","odoroki","takaburi","haji",]

#本の感情をベクトルに変換
def change_vector(dic):
  vector = []
  for feature in features:
    vector.append(dic[feature])
  return vector

#本の感情分析
def sentiment_analyze(outline):
  sentiment_dic = {}
  result = analyzer.analyze(outline)
  emotion = result["emotion"]
  if emotion != None:
    for feature in features:
      if emotion[feature] == None:
        sentiment_dic[feature] == 0
      else:
        sentiment_dic[feature] = len(emotion[feature])
    book_vector = change_vector(sentiment_dic)
  else:
    book_vector = [0,2,4,1,1,0,0,0,2,1]
  return book_vector

#本とカテゴリの類似度計算
def calulate_cos(book,category):
  x = np.dot(book,category)
  a = math.sqrt(sum(list(map(lambda x: x**2,book)))+0.01)
  b = math.sqrt(sum(list(map(lambda x: x**2,category)))+0.01)
  category_vector = x / (a*b)
  return category_vector


@app.route('/')
def hello_world():
    ver = sys.version
    # target = os.environ.get('TARGET', 'World')
    return render_template('index.html')

# "/" →　"〇〇.html"の結果表示のとこへ変更する
@app.route("/result.html",methods=["POST"])
def show():
    title = request.form["title"]
    #楽天APIにタイトル名を渡す.
    title, url, book_large_image, book_medium_image, book_small_image, author, review, price_yen, publish_name, item_caption = get_books_by_title(title)
    #感情分析を行う．
    if item_caption != None:
      book_vector = sentiment_analyze(item_caption)
    else: 
      book_vector = [0,2,4,1,1,0,0,0,2,1]
    #本とカテゴリの類似度計算
    cos = 0
    playlist_id = ""
    for id, music_vector in RECOMMEND_PLAYLIST.items():
        if cos < calulate_cos(book_vector,music_vector):
            cos = calulate_cos(book_vector,music_vector)
            playlist_id = id
    #spotify APIにplaylist_idを渡す
    music_data = get_songs_from_playlist(playlist_id)
    song_name_1, artist_1, ref_1, music_image_1 = music_data[0]  
    song_name_2, artist_2, ref_2, music_image_2 = music_data[1] 
    song_name_3, artist_3, ref_3, music_image_3 = music_data[2] 
    song_name_4, artist_4, ref_4, music_image_4 = music_data[3] 
    return render_template('result.html',
    title=title,url=url,book_large_image=book_large_image,book_medium_image=book_medium_image,book_small_image=book_small_image,author=author,review=review,price_yen=price_yen,publish_name=publish_name,
    song_name_1=song_name_1,artist_1=artist_1,ref_1=ref_1,music_image_1=music_image_1,
    song_name_2=song_name_2,artist_2=artist_2,ref_2=ref_2,music_image_2=music_image_2,
    song_name_3=song_name_3,artist_3=artist_3,ref_3=ref_3,music_image_3=music_image_3,
    song_name_4=song_name_4,artist_4=artist_4,ref_4=ref_4,music_image_4=music_image_4,)

@app.route('/about')
def aboutPage():
    return render_template('about.html')

def get_songs_from_playlist(playlist_id: str):
    songs = []
    result = spotify.playlist(playlist_id)
    for item in result['tracks']['items']:
        song_name = item['track']['name']
        if len(item['track']['artists']) == 0:
          artist = None
        else:
          artist = item['track']['artists'][0]['name']
        ref = item['track']['external_urls']['spotify']
        if len(item['track']['album']['images']) <= 1:
          medium_image = None
        else:
          medium_image = item['track']['album']['images'][1]['url']
        # large_image = item['track']['album']['images'][0]['url']
        # small_image = item['track']['album']['images'][2]['url']
        # preview_url = item['track']['preview_url']
        songs.append([song_name, artist, ref, medium_image]) # 曲名, アーティスト, URL, 画像(中)
    if len(songs) < RECOMMEND_NUM:
        return songs
    else:
        return random.sample(songs, RECOMMEND_NUM)

def get_books_by_title(title: str):
    url = "https://app.rakuten.co.jp/services/api/Kobo/EbookSearch/20170426"
    params = {
                "format": "json",
                "language": "JA",
                "applicationId": APP_ID,
                "title": title
             }
    r = requests.get(url, params=params)
    # とりあえず1つだけ返す
    for item in r.json()['Items']:
        title = item['Item']['title']
        url = item['Item']['itemUrl']
        large_image = item['Item']['largeImageUrl']
        medium_image = item['Item']['mediumImageUrl']
        small_image = item['Item']['smallImageUrl']
        author = item['Item']['author']
        review = item['Item']['reviewAverage']
        price_yen = item['Item']['itemPrice']
        publish_name = item['Item']['publisherName']
        item_caption = item['Item']['itemCaption']
        return [title, url, large_image, medium_image,small_image, author, review, price_yen, publish_name, item_caption] # タイトル, URL，画像, 作者, 評価(5点満点), 価格(円)，レーベル，あらすじ
    return [None,None,None,None,None,None,None,None,None,None]# 検索結果なし


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8888)))




@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/img/favicon_io'), 'favicon.ico', )