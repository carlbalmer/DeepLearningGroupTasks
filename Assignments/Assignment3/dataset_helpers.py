import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class DogsDataset(Dataset):
    """Dataset
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        # print(tmp_df)
        # assert tmp_df['breed'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
        # "Some images referenced in the CSV file were not found"

        self.mlb = LabelEncoder()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['id']
        self.y_train = self.mlb.fit_transform(tmp_df['breed'])  # having problem shaping it in the right size

        # print(self.y_train[0])

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = (self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


def make_stratified_splits(dataset):
    x = dataset.X_train
    y = dataset.y_train
    test_straf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=4456)

    train_straf = StratifiedShuffleSplit(n_splits=1, test_size=0.125, train_size=0.875, random_state=58778)
    rest_index, test_index = next(test_straf.split(x, y))
    # print("rest:", X[rest_index], "\nTEST:", X[test_index])

    train_index, val_index = next(train_straf.split(x[rest_index], y[rest_index]))
    # print("train:", X[train_index], "\nval:", X[val_index])

    # we can equiv also retrn these indexes for the random sampler to do its job
    # print(test_index,train_index,val_index)
    return train_index, val_index, test_index
