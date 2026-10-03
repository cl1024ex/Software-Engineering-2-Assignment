import pandas as pd

database = pd.read_csv("data/database.csv")

df = pd.DataFrame(database)

print(df.head())
