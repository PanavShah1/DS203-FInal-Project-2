import requests

API_KEY = "AIzaSyBqcOYR6S4xv_-iun3xQFStdeY35v5F-zE"
CHANNEL_ID = "UC5OrDvL9DscpcAstz7JnQGA"  # Michael Jackson's channel ID

def get_all_videos(channel_id, api_key):
    videos = []
    next_page_token = None

    # Loop to fetch all videos from the channel
    while True:
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={channel_id}&key={api_key}&maxResults=50&order=date"
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching videos. Status code: {response.status_code}")
            print("Response:", response.text)  # Print error message for debugging
            return []

        data = response.json()
        for item in data['items']:
            # Only process 'video' type items
            if item['id']['kind'] == 'youtube#video':
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                link = f"https://www.youtube.com/watch?v={video_id}"
                videos.append({'title': title, 'link': link})

        # Check if there are more pages to fetch
        next_page_token = data.get('nextPageToken')
        if not next_page_token:
            break

    return videos

def print_video_list(videos):
    if not videos:
        print("No videos found.")
        return

    # Print out the list of videos
    for video in videos:
        print(f"Title: {video['title']}")
        print(f"Link: {video['link']}")
        print("-" * 50)

# Fetch all videos from Michael Jackson's channel
videos = get_all_videos(CHANNEL_ID, API_KEY)
print_video_list(videos)
