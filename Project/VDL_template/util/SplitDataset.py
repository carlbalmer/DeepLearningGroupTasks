import numpy
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
PATH = "Data/"
data = pd.read_csv(PATH+"trainLabels.csv")

first_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=4456)
second_split = StratifiedShuffleSplit(n_splits=1, test_size=0.125, train_size=0.875, random_state=58778)

rest_index, test_index = next(first_split.split(numpy.zeros(len(data['level'])), data['level']))
rest_data, test_data = data.iloc[rest_index], data.iloc[test_index]

train_index, val_index = next(second_split.split(numpy.zeros(len(rest_data['level'])), rest_data['level']))
train_data, val_data = rest_data.iloc[train_index], rest_data.iloc[val_index]

train_data.to_csv(PATH+'train.csv')
val_data.to_csv(PATH+'val.csv')
test_data.to_csv(PATH+'test.csv')