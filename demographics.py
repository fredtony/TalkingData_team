import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle

with open('combined_df.pkl', 'r') as f:
    df = pd.DataFrame(pickle.load(f))
    
