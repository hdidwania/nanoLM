import argparse

import torch

import tokenizer as tokenizer_
from model import LanguageModel

MAXLEN = 512
MINFREQ = 0
LOAD_PATH = "saved-models/model-checkpoint-1000.pt"

tokenizer_class_dict = {
    "word": tokenizer_.WordTokenizer,
    "char": tokenizer_.CharTokenizer,
}


def decode_next(logits):
    output_token = torch.argmax(logits).item()
    return output_token


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Preparing Tokenizer")
    tokenizer = tokenizer_class_dict[args.tokenizer](
        maxlen=args.maxlen, minfreq=args.minfreq, path=args.tokenizer_path
    )
    # tokenizer = tokenizer_.CharTokenizer(
    #     maxlen=MAXLEN, minfreq=MINFREQ, path="./saved-models/tokenizer-char-512-0.pkl"
    # )
    # tokenizer.load(f"saved-models/tokenizer-{MAXLEN}-{MINFREQ}.pkl")
    print("Preparing Model")
    model = LanguageModel(
        dims=args.dims,
        heads=args.heads,
        nblocks=args.nblocks,
        vocab_size=len(tokenizer.vocab),
        maxlen=MAXLEN,
        padding_idx=tokenizer.token_to_idx["<PAD>"],
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
        if output_token == tokenizer.token_to_idx[tokenizer.EOS_TOKEN]:
            break
    print(tokenizer.decode(tokens))

    # TODO: Peplexity
    # TODO: Check why outputs are not deterministic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--nblocks", type=int, default=3)
    parser.add_argument(
        "--tokenizer", help="Type of tokenizer", choices=["word", "char"], required=True
    )
    parser.add_argument(
        "--tokenizer_path",
        help="Path of pickle file for loading the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--maxlen", help="Maximum length of sentence", required=True, type=int
    )
    parser.add_argument(
        "--minfreq", help="Minimum freq of words to retain", default=0, type=int
    )

    args = parser.parse_args()
    main(args)
