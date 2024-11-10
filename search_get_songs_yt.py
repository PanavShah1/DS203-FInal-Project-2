import yt_dlp

def download_audio(search_query, folder_name):
    # Define options for yt-dlp to download the best audio and convert it to .wav
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{folder_name}/%(title)s.%(ext)s",  # Save with song title as filename
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Set output format to WAV
            'preferredquality': '192',
        }],
        'noplaylist': True,  # Ensure we are not downloading playlists
        'quiet': False,  # Set to False to show more verbose logs if needed
        'extract_flat': False,  # Fetch video metadata as well, not just URLs
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Get the search results for the song from YouTube (limit to top results)
        result = ydl.extract_info(f"ytsearch:{search_query}", download=False)
        
        # Loop through the top 30 results
        for idx, entry in enumerate(result['entries'][:30], 1):  # Limit to top 30 results
            video_url = entry['url']
            video_duration = entry['duration']  # Duration in seconds
            
            # Check if the video duration is less than 5 minutes (300 seconds)
            if video_duration < 300:
                print(f"Downloading {idx}: {entry['title']} from {video_url}")
                # Download the audio of the video
                ydl.download([video_url])  # Download the audio from the URL
                print(f"Song {idx} downloaded.")
            else:
                print(f"Skipping {entry['title']} as it is longer than 5 minutes.")

# Example usage for searching the Indian National Anthem on YouTube
search_query = "Indian National Anthem"
folder_name = 'songs.nosync/Indian_National_Anthem'
download_audio(search_query, folder_name)
