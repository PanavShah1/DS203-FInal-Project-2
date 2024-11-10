import yt_dlp
import pandas as pd
import numpy as np

def download_audio_from_youtube(links, folder_name, max_duration=600):
    # Define options for yt-dlp to download the best audio and convert it to .wav
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{folder_name}/%(title)s.%(ext)s",  # Save with song title as filename
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Set output format to WAV
            'preferredquality': '192',
        }],
        'quiet': False,  # Set to False to show more verbose logs if needed
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for idx, video_url in enumerate(links, 1):  # Iterate through the list of links
            # Extract info about the video
            video_info = ydl.extract_info(video_url, download=False)
            video_duration = video_info.get('duration', 0)  # Get duration in seconds
            
            # Check if the video duration is less than 10 minutes (600 seconds)
            if video_duration < max_duration:
                print(f"Downloading {idx}: {video_url} ({video_duration / 60:.2f} mins)")
                ydl.download([video_url])  # Download the audio from the URL
                print(f"Song {idx} downloaded.")
            else:
                print(f"Skipping {idx}: {video_url} - Duration {video_duration / 60:.2f} mins (longer than 10 minutes).")

# Assuming you already have the list of YouTube links stored in `links`
# Example usage:

df = pd.read_csv('list_songs_sheet/Marathi_Bhavgeet - Sheet1.csv', header=None, index_col=False)
links = df[0].to_list()

folder_name = 'songs.nosync/Marathi_Bhavgeet'
download_audio_from_youtube(links, folder_name)
