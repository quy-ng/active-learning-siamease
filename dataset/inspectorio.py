import pandas as pd
from torch.utils.data import Dataset

from dataset.augmentation import augment


class Inspectorio(Dataset):

    def __init__(self, file_path, transform=None):
        df = pd.read_excel(file_path)
        augmented_df = transform(df)

        self.data = augmented_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name = self.data.iloc[index, 0]
        address = self.data.iloc[index, 1]
        return name, address


if __name__ == '__main__':
    foo = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment)
    for i in range(len(foo)):
        sample = foo[i]
        print(sample)
        break
    print('#' * 10)
    print(foo[1:3])
