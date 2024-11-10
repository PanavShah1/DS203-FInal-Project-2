from get_songs_spotify import get_artist_id, get_tracks_from_playlists
from download_songs import get_track_info_from_spotify, download_audio

access_token = "BQDBN8S31x43ydk5q2SsPGrhhrrOaCduDLdJpULyM8MYs-xIeRnP9A8ybuZY5hNJtwXRY_PUA5JZH03FsdAqf-hgk1OpIOXs1Fr3pMoT88aUoMCeSG8"  # Replace with your valid token
artist_name = "Kishore Kumar"
folder_name = 'Kishore_Kumar'
n = 100

links = []
artist_id = get_artist_id(artist_name, access_token)
if artist_id:
    tracks = get_tracks_from_playlists(artist_id, access_token, n)
    for idx, track in enumerate(tracks, 1):
        print(f"{idx}. {track['name']} - {track['link']}")
        links.append(track['link'])

print(links)


for i in range(3, 100):
    spotify_url = links[i]
    get_track_info_from_spotify(spotify_url, f"songs.nosync/{folder_name}")
    print()
    print(f"Song {i} downloaded.")
    print()

