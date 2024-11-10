import requests
import base64

API_KEY = "AIzaSyBqcOYR6S4xv_-iun3xQFStdeY35v5F-zE"

def get_channel_id(channel_name):
    response = requests.get(f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={channel_name}&key={API_KEY}")
    if response.status_code == 200:
        data = response.json()
        channel_id = data["items"][0]["snippet"]["channelId"]
        return channel_id
    else:
        print(f"Error: {response.status_code}")
        return None

def get_top_videos(CHANNEL_ID):
    response = requests.get(f"https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={CHANNEL_ID}&key={API_KEY}&maxResults={50}&order=viewcount")
    if response.status_code == 200:
        data = response.json()
        data_list = list(data['items'])
        
        for video in data_list:
            if video['id']['kind'] == 'youtube#video':  # Ensure it's a video
                video_id = video['id']['videoId']
                title = video['snippet']['title']
                link = f"https://www.youtube.com/watch?v={video_id}"
                print(f"Title: {title}\nLink: {link}")
    else:
        print(f"Error: {response.status_code}")
        return None

get_top_videos(get_channel_id("Michael Jackson"))
