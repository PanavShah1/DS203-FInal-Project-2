import requests

# curl -X POST "https://accounts.spotify.com/api/token" \
#      -H "Authorization: Basic $(echo -n 'f5fdbbc442964397822f07a079df3cc0:6a1ed9c4f211470192b8a0afeb93b1b3' | base64)" \
#      -d "grant_type=client_credentials"



# Get Artist ID
def get_artist_id(artist_name, access_token):
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": artist_name,
        "type": "artist",
        "limit": 1  # Get only the top match
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["artists"]["items"]:
            return data["artists"]["items"][0]["id"]
        else:
            print("Artist not found.")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

# Get Top 100 Tracks by Searching for Playlists (using pagination)
def get_tracks_from_playlists(artist_id, access_token, n):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "limit": 50  # Limit to 50 albums per request
    }
    tracks = []
    
    while len(tracks) < n:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            albums = response.json()["items"]
            for album in albums:
                album_id = album["id"]
                # Get tracks from this album
                album_tracks_url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
                album_tracks_response = requests.get(album_tracks_url, headers=headers)
                if album_tracks_response.status_code == 200:
                    tracks_data = album_tracks_response.json()["items"]
                    for track in tracks_data:
                        tracks.append({
                            "name": track["name"],
                            "link": track["external_urls"]["spotify"]
                        })
                if len(tracks) >= n:
                    break
            params["offset"] = len(albums)
        else:
            print(f"Error: {response.status_code}")
            break
    
    return tracks[:n]

if __name__ == "__main__":
    access_token = "BQBDJnLJo7wn4vXVUFTG5dzpoZE94xMSdkgTyqRZm9L-ZSWbSVf84HrXNbMxy2MCx4yI3Jvc2-AtxIrDyBYMnQYs6rmpYPFh5c_i0jgp5-myeZuZddc"  # Replace with your valid token
    artist_name = "Asha Bhosle"
    n = 100

    artist_id = get_artist_id(artist_name, access_token)
    if artist_id:
        tracks = get_tracks_from_playlists(artist_id, access_token, n)
        for idx, track in enumerate(tracks, 1):
            print(f"{idx}. {track['name']} - {track['link']}")
