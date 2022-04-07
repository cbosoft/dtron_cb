from torch.utils.data import Dataset as _Dataset, DataLoader


class Dataset(_Dataset):

    def __init__(self, data, mapper):
        self.mapper = mapper
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.mapper(self.data[item])
