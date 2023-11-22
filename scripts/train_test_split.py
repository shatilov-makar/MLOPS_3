from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('/home/makar/MLops_3/datasets/data_processed.csv', index_col=0)
Train, Test = train_test_split(df, test_size=0.2, random_state=42)

Train.to_csv('/home/makar/MLops_3/datasets/data_train.csv', index=0)
Test.to_csv('/home/makar/MLops_3/datasets/data_test.csv', index=0)
