import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp

def get_track_info_from_spotify(url, folder):
    # Set up the Spotipy client
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f5fdbbc442964397822f07a079df3cc0", client_secret="6a1ed9c4f211470192b8a0afeb93b1b3"))
    
    # Extract track ID from URL
    track_id = url.split('/')[-1].split('?')[0]
    track = sp.track(track_id)
    
    # Extract track metadata
    track_name = track['name']
    artist_name = track['artists'][0]['name']
    track_url = track['external_urls']['spotify']
    
    print(f"Track Name: {track_name}")
    print(f"Artist: {artist_name}")
    print(f"Track URL: {track_url}")
    
    # Convert track details to a YouTube search URL
    youtube_search_query = f"{track_name} {artist_name}"
    youtube_search_url = f"https://www.youtube.com/results?search_query={youtube_search_query}"
    print(f"YouTube Search URL: {youtube_search_url}")
    
    # Download audio from the topmost result on YouTube
    download_audio(youtube_search_query, track_name, artist_name, folder)

def download_audio(search_query, track_name, artist_name, folder):
    # Define options for yt-dlp to download best audio and convert it to .wav
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{folder}/{track_name}.%(ext)s",  # Save file with the track name
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Set output format to WAV
            'preferredquality': '192',
        }],
        'noplaylist': True,  # Ensure only the first result is downloaded
        'quiet': True,  # Reduce verbosity
        'extract_flat': True,  # Only fetch the video URLs (don't fetch metadata)
        # 'max_downloads': 1,  # Limit the download to the first video only
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Get the search results for the song from YouTube (limit to top result)
        result = ydl.extract_info(f"ytsearch:{search_query}", download=False)
        
        # Ensure we are selecting the topmost result (first in the list)
        video_url = result['entries'][0]['url']  # Topmost (first) video URL
        video_duration = result['entries'][0]['duration']  # Duration in seconds
        
        # Check if the video duration is less than 10 minutes (600 seconds)
        if video_duration < 600:
            print(f"Downloading from: {video_url}")
            # Download the audio of the topmost result
            ydl.download([video_url])  # Download the audio from the URL
            print(f"Audio file saved as '{track_name}.wav'")
        else:
            print(f"Skipping download. The video is longer than 10 minutes ({video_duration // 60} minutes).")

if __name__ == "__main__":
    spotify_url = "https://open.spotify.com/track/3QGsuHI8jO1Rx4JWLUh9jd"  # Replace with actual track URL
    get_track_info_from_spotify(spotify_url, "songs")
    get_track_info_from_spotify("https://open.spotify.com/track/4zrKN5Sv8JS5mqnbVcsul7?si=e0ca510a338544da", "songs")

