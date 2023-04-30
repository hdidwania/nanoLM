import argparse
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import MovieData
from model import LanguageModel
from tokenizer import Tokenizer

MAXLEN = 128
MINFREQ = 5
LOAD_PATH = "saved-models/model-checkpoint-1000.pt"


def decode_next(logits):
    output_token = torch.argmax(logits).item()
    return output_token

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Preparing Tokenizer")
    tokenizer = Tokenizer(maxlen=MAXLEN, minfreq=MINFREQ)
    tokenizer.load(f"saved-models/tokenizer-{MAXLEN}-{MINFREQ}.pkl")
    print("Preparing Model")
    model = LanguageModel(
        dims=args.dims,
        heads=args.heads,
        nblocks=args.nblocks,
        vocab_size=len(tokenizer.vocab),
        maxlen=MAXLEN,
        padding_idx=tokenizer.word_to_idx["<PAD>"],
        device=device,
    )
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()

    context_text = input("Enter Context: ")
    tokens = tokenizer.encode(context_text)
    curr_index = len(tokens)
    while len(tokens) < 100:
        curr_index += 1
        tokens_input = torch.tensor([tokens]).to(device)
        output_logits = model(tokens_input)
        output_token = decode_next(output_logits[0][-1])
        tokens.append(output_token)
        if output_token == tokenizer.word_to_idx[tokenizer.EOS_TOKEN]:
            break
    print(tokenizer.decode(tokens))

    # TODO: Peplexity
    # TODO: Check why outputs are not deterministic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--nblocks", type=int, default=3)
    args = parser.parse_args()
    main(args)
