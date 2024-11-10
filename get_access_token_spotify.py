import requests
import base64

def get_access_token(client_id, client_secret):
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
    }

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info["access_token"]
    else:
        print(f"Error: {response.status_code}, {response.json()}")
        return None

# Replace with your actual Client ID and Client Secret
client_id = "f5fdbbc442964397822f07a079df3cc0"
client_secret = "6a1ed9c4f211470192b8a0afeb93b1b3"

access_token = get_access_token(client_id, client_secret)
print(f"Access Token: {access_token}")
