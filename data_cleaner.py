import pandas as pd

# load data
df = pd.read_csv('data/data_raw/data.csv')

df_new = df[df['year'] >= 1990]  # keep songs from 1990 onwards
df_new = df_new.drop_duplicates(subset=['name', 'artists'])  # drop duplicates by name and artists

# save cleaned data
df_new.to_csv('data/data_cleaned.csv', index=False)