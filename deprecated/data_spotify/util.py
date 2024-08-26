import requests
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import html
import pandas as pd
import threading
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import time
from collections import defaultdict
import multiprocessing


def add_artists_to_adjacency(A, item, ids_to_ind):
    artists = item["artists"]
    if len(artists) < 2:
        return

    master_ids = ids_to_ind.keys()

    ids = [s["id"] for s in artists if s["id"] in master_ids]

    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue
            A[ids_to_ind[ids[i]]][ids_to_ind[ids[j]]] = 1


def get_map_and_numpy_array(df):
    id_map = pd.Series(df.index, index=df["id"]).to_dict()
    name_map = pd.Series(df.index, index=df["name"]).to_dict()

    with open("assets/id_to_ind.json", "w") as file:
        json.dump(id_map, file, indent=2)

    with open("assets/name_to_ind.json", "w") as file:
        json.dump(name_map, file, indent=2)

    mat = df.iloc[:, 2:].reset_index(drop=True).to_numpy()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(mat)

    np.save("assets/X.npy", X)

    return X, id_map, name_map


def get_name_id_map():
    ids, names, back = {}, {}, defaultdict(list)
    with open("assets/id_to_ind.json", "r") as file:
        ids = json.load(file)
    with open("assets/name_to_ind.json", "r") as file:
        names = json.load(file)
    for k, v in ids.items():
        back[v].append(k)
    for k, v in names.items():
        back[v].append(k)
    return ids, names, back


def get_client():
    load_dotenv(dotenv_path=".env")
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    auth_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret, requests_timeout=5
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp


def get_artists_and_genres(preloaded=False):
    if preloaded:
        ids, genres = [], []
        with open("assets/artists.json", "r") as file:
            ids = json.load(file)
        with open("assets/genres.json", "r") as file:
            genres = json.load(file)
        return ids, genres

    artists_html = requests.get("https://kworb.net/spotify/listeners.html").text
    genres_html = requests.get("https://www.everynoise.com/everynoise1d.html").text

    # get artist ids
    pattern = r'href="artist/([^"]+)\.html"'
    artistIds = re.findall(pattern, artists_html)
    for i in range(len(artistIds)):
        artistIds[i] = re.sub(r"_songs", "", artistIds[i])

    with open("assets/artists.json", "w") as file:
        json.dump(artistIds, file, indent=2)

    # get genres
    pattern = r'>([^"]+)</a></td>'
    genres = re.findall(pattern, genres_html)
    decoded_genres = [html.unescape(genre) for genre in genres if "&#x260A;" != genre]

    with open("assets/genres.json", "w") as file:
        json.dump(decoded_genres, file, indent=2)

    print(len(artistIds))
    print(len(decoded_genres))

    return artistIds, decoded_genres


def get_artists_df(sp, artists, decoded_genres, size=50):
    headers, num_genres = ["id", "name", "followers", "popularity"] + decoded_genres + [
        ""
    ] * 2500, len(decoded_genres) + 2500

    genres_to_ind = {decoded_genres[i]: i + 4 for i in range(len(decoded_genres))}
    genres_set = set(decoded_genres)

    df = pd.DataFrame(columns=headers)
    lock = threading.Lock()

    failed = []

    def get_artist_info(artists):
        nonlocal df
        try:
            group_data = sp.artists(artists)

            with lock:
                for data in group_data["artists"]:
                    row = [data["id"]]
                    row.append(data["name"])
                    print("Adding: ", data["name"])
                    row.append(data["followers"]["total"])
                    row.append(data["popularity"])
                    row = row + [0] * num_genres
                    genres = data["genres"]
                    for g in genres:
                        if g not in genres_set:
                            genres_set.add(g)
                            genres_to_ind[g] = len(genres_set)

                        row[genres_to_ind[g]] = 1
                    df.loc[len(df)] = row
        except BaseException as e:
            with lock:
                print(e)
                failed.append(artists)

    threads = []
    for i in range(0, len(artists) // size):
        thread = threading.Thread(
            target=get_artist_info, args=(artists[size * i : size * (i + 1)],)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    while failed:
        time.sleep(30)
        f = failed.pop()
        get_artist_info(f)

    filtered_df = df.loc[:, df.sum() != 0]

    return filtered_df


def get_artist_albums_and_collabs(aid, A, albums_seen, sp, ids_to_ind, limit=50):
    print("starting ", aid)
    try:
        lookup_set = set()
        lookup_list = []
        items = []

        resp = sp.artist_albums(
            artist_id=aid,
            album_type="single,appears_on,compilation,album",
            limit=limit,
        )

        print(resp)

        items = items + resp["items"]

        while resp:
            print("got resp response for ", aid)
            resp = sp.next(resp)
            if resp is not None:
                items = items + resp["items"]

        for item in items:
            tid = item["id"]
            album_type = item["album_type"]
            if albums_seen[tid] or tid in lookup_set or album_type == "compilation":
                continue
            albums_seen.add(tid)
            lookup_set.add(tid)

            total_tracks = item["total_tracks"]
            if total_tracks == 1:
                add_artists_to_adjacency(A, item, ids_to_ind)
                continue

            lookup_list.append(tid)

        all_tracks = []
        chunk_size = 20

        for i in range(0, len(lookup_list), chunk_size):
            chunk = lookup_list[i : i + chunk_size]
            print(aid, chunk)
            out = sp.albums(chunk)
            # print(json.dumps(out))

            # tracks = get_tracks
            for album in out["albums"]:
                all_tracks = all_tracks + album["tracks"]["items"]

        for track in all_tracks:
            add_artists_to_adjacency(A, track, ids_to_ind)
    except BaseException as e:
        print(aid, e)
        time.sleep(15)
        get_artist_albums_and_collabs(aid)
        # with lock:
        #     print(aid, e)
        #     failed.append(aid)


def get_adjacency_matrix(sp, artists, ids_to_ind):
    n = len(artists)
    # A = np.zeros((n, n))

    # albums_seen = set()
    # lock = threading.Lock()
    # failed = []

    manager = multiprocessing.Manager()

    # Using manager to create shared memory objects
    A = manager.list([manager.list([0] * n) for _ in range(n)])
    albums_seen = manager.dict()

    threads = []
    for i in range(0, len(artists)):
        if i != 0 and i % 100 == 0:
            time.sleep(15)
            print(
                "+++++++++++++++++++++++++++++++", i, "+++++++++++++++++++++++++++++++"
            )
        thread = multiprocessing.Process(
            target=get_artist_albums_and_collabs, args=(artists[i], A, albums_seen, sp, ids_to_ind)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # while failed:
    #     time.sleep(30)
    #     f = failed.pop()
    #     get_artist_albums_and_collabs(f)

    np.save("assets/A.npy", np.array(A))

    return A
