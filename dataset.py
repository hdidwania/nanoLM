import pandas as pd
from torch.utils.data import Dataset


class MovieData(Dataset):
    def __init__(self, maxlen=256, train_size=0.9, split="train"):
        self.maxlen = maxlen  # Maximum length of sentences

        df = pd.read_csv("../data/wiki_movie_plots.csv")
        self.data = list(df["Plot"])
        # Only keep samples which are shorter than maxlen
        self.data = [s for s in self.data if len(s.split(" ")) <= self.maxlen]
        train_size = int(train_size * len(self.data))
        if split == "train":
            self.data = self.data[:train_size]
        elif split == "val":
            self.data = self.data[train_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
