import argparse

import tokenizer
from dataset import MovieData

tokenizer_class_dict = {
    "word": tokenizer.WordTokenizer,
    "char": tokenizer.CharTokenizer,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", help="Type of tokenizer", choices=["word", "char"], required=True
    )
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
    tokenizer = tokenizer_class_dict[args.tokenizer](args.maxlen, args.minfreq)
    tokenizer.create_vocab(data.data)

    tokenizer.save(args.savepath)
