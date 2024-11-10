import pandas as pd
import shutil
import os

# Load the CSV file
df = pd.read_csv('DS203-audio-labels - Sheet1 (4).csv')
print(df.head())

# Replace TRUE/FALSE with 1/0
df.replace('TRUE', 1, inplace=True)
df.replace('FALSE', 0, inplace=True)

# Drop unnecessary columns
df.drop(['Done', 'Song Duration', 'Youtube link', 'Spotify link'], axis=1, inplace=True)
print(df.columns.values)

# Define the folder and artist mappings
artists_folders = ['Asha_Bhosle', 'Michael_Jackson', 'Kishore_Kumar', 'Indian_National_Anthem', 'Marathi_Bhavgeet', 'Marathi_Lavani']
artist_columns = ['Asha Bhosale', 'Michael Jackson', 'Kishor Kumar', 'National Anthem', "Marathi ‘Bhav Geet’", 'Marathi Lavni']

# Create folders for each artist in 'fake/' if they don't exist
# for folder in artists_folders:
#     os.makedirs(f'fake/{folder}', exist_ok=True)

# Loop through each song and assign it to the appropriate artist
for i in range(len(df)):
    artist_folder = None
    for j, column in enumerate(artist_columns):
        if df.iloc[i][column] == 1:  # Check if the song belongs to this artist
            artist_folder = artists_folders[j]
            break

    print(artist_folder)  # Verify if the correct artist folder is identified

    if artist_folder:
        file = f'MFCC_T/{i+1:02d}-MFCC.csv'  # Assuming song numbers are 1-indexed
        target_path = f'fake/{artist_folder}/{i+1:02d}-MFCC.csv'
        
        # Copy the file if it exists
        try:
            shutil.copyfile(file, target_path)
        except FileNotFoundError:
            print(f"File {file} not found.")
