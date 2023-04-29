import argparse

from dataset import MovieData
from tokenizer import Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maxlen", help="Maximum length of sentence", required=True, type=int
    )
    parser.add_argument(
        "--minfreq", help="Minimum freq of words to retain", default=0, type=int
    )
    parser.add_argument(
        "--savepath",
        help="Path to pickle file for saving the tokenizer",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    data = MovieData(args.maxlen)
    tokenizer = Tokenizer(args.maxlen, args.minfreq)
    tokenizer.create_vocab(data.data)

    tokenizer.save(args.savepath)
