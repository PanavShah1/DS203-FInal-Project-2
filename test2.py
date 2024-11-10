import yt_dlp

def download_audio(url, output_directory):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{output_directory}/audio.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Set output format to WAV
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Audio file saved in {output_directory} as 'audio.wav'")

# Example usage
url = "https://www.youtube.com/watch?v=JGwWNGJdvx8"
output_directory = "."
download_audio(url, output_directory)
