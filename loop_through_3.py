import os
import yt_dlp
import pandas as pd
from tqdm import tqdm


def download_audio_from_playlist(playlist_url, folder_name, max_duration=600):
    # Ensure the folder exists, create it if necessary
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"Folder already exists: {folder_name}")
    
    # Define options for yt-dlp to download the best audio and convert it to .wav
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(folder_name, '%(title)s.%(ext)s'),  # Save with song title as filename in the correct folder
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Set output format to WAV
            'preferredquality': '192',
        }],
        'quiet': False,  # Set to False to show more verbose logs if needed
    }
    
    # Initialize yt-dlp with the given options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info about the playlist
        playlist_info = ydl.extract_info(playlist_url, download=False)
        
        # Get the entries in the playlist
        for idx, entry in enumerate(playlist_info['entries'], 1):
            video_url = entry['url']
            video_duration = entry['duration']  # Duration in seconds
            
            # Check if the video duration is less than 10 minutes (600 seconds)
            if video_duration < max_duration:
                print(f"Downloading {idx}: {entry['title']} from {video_url}")
                ydl.download([video_url])  # Download the audio from the URL
                print(f"Song {idx} downloaded.")
            else:
                print(f"Skipping {entry['title']} as it is longer than {max_duration//60} minutes.")

# Example usage:
playlist_url = "https://www.youtube.com/playlist?list=PLGpLqaMxppFq6htuAYtoKIlwItU_Gl_Zs"
folder_name = 'songs.nosync/Marathi_Bhavgeet'

download_audio_from_playlist(playlist_url, folder_name)
