import pandas as pd
df = pd.read_csv('KDDTest-21.csv')
df.to_parquet('KDDTest-21.parquet')