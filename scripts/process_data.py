import pandas as pd
import numpy as np

df = pd.read_csv('/home/makar/MLops_3/datasets/data.csv', index_col=0)

df = df.drop(df[df['num_sold'] > 915].index)
df['date'] = pd.to_datetime(df['date'])
df.insert(loc=2, column='year', value=df['date'].dt.year)
df.insert(loc=3, column='month', value=df['date'].dt.month)
df.insert(loc=4, column='day', value=df['date'].dt.day)
df = df.drop(['date', 'row_id'], axis=1)
df = df.reset_index(drop=True)

df['num_sold'] = np.log(df['num_sold'])
df['year'] = df['year'].astype('category')
df['month'] = df['month'].astype('category')
df['day'] = df['day'].astype('category')
df = pd.get_dummies(df)

df.to_csv('/home/makar/MLops_3/datasets/data_processed.csv', index='Row_id')
