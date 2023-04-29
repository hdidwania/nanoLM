import pandas as pd
from torch.utils.data import Dataset


class MovieData(Dataset):
    def __init__(self, maxlen=256):
        self.maxlen = maxlen  # Maximum length of sentences

        df = pd.read_csv("../data/wiki_movie_plots.csv")
        self.data = list(df["Plot"])
        # Only keep samples which are shorter than maxlen
        self.data = [s for s in self.data if len(s.split(" ")) <= self.maxlen]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
