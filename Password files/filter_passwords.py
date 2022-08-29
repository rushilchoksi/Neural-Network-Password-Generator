import pandas as pd

df = pd.read_csv('rock.csv')
df['pass'] = df['pass'].astype('str')
mask = (df['pass'].str.len() == 10)
df = df.loc[mask]
df.to_csv("asd.csv")
# print(df)