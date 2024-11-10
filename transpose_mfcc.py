import numpy as np
import pandas as pd
from tqdm import tqdm

def transpose_mfcc(mfcc_file, output_file):
    # Load the MFCC file
    mfcc_df = pd.read_csv(mfcc_file, header=None, index_col=False)
    
    # Transpose the MFCC coefficients
    mfcc_transposed = mfcc_df.T
    
    # Save the transposed MFCC coefficients to a new file
    mfcc_transposed.to_csv(output_file, header=False, index=False)


for i in tqdm(range(1, 117)):
    transpose_mfcc(f"MFCC-files/{i:02d}-MFCC.csv", f"MFCC_T/{i:02d}-MFCC.csv")