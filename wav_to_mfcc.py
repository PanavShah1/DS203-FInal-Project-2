from create_mfcc_coefficients import create_MFCC_coefficients
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

SONG_DIR = Path("songs.nosync")
GET_FOLDER = SONG_DIR / "Marathi_Bhavgeet"
MFCC_DIR = Path("mfcc.nosync")
OUTPUT_FOLDER = MFCC_DIR / "Marathi_Bhavgeet"

if not os.path.exists(OUTPUT_FOLDER):
    print("making dir")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

songs_wav = os.listdir(GET_FOLDER)

for song in tqdm(songs_wav):
    song_path = GET_FOLDER / song
    mfcc_coefficients = create_MFCC_coefficients(song_path)
    
    mfcc_df = pd.DataFrame(mfcc_coefficients)
    
    output_file = OUTPUT_FOLDER / f"{song.replace('.wav', '')}_mfcc.csv"
    mfcc_df = mfcc_df.T
    mfcc_df.to_csv(output_file, header=False, index=False)

