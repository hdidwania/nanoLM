import argparse

import torch

import tokenizer as tokenizer_
from model import LanguageModel

MAXLEN = 512
MINFREQ = 0

tokenizer_class_dict = {
    "word": tokenizer_.WordTokenizer,
    "char": tokenizer_.CharTokenizer,
}


def decode_next(logits, avoid_unk=False):
    if not avoid_unk:
        output_token = torch.argmax(logits).item()
    else:
        output_token = torch.topk(logits, k=2)
        output_token = output_token.indices.cpu().numpy().tolist()
    return output_token


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Preparing Tokenizer")
    tokenizer = tokenizer_class_dict[args.tokenizer](
        maxlen=args.maxlen, minfreq=args.minfreq, path=args.tokenizer_path
    )
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
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    context_text = input("Enter Context: ")
    tokens, pad_mask = tokenizer.encode(context_text, return_pad_mask=True)
    curr_index = len(tokens)
    while len(tokens) < 100:
        curr_index += 1
        tokens_input = torch.tensor([tokens]).to(device)
        pad_mask_input = torch.tensor([pad_mask]).to(device)
        output_logits = model(tokens_input, pad_mask_input)
        output_token = decode_next(output_logits[0][-1], avoid_unk=True)
        if output_token[0] == tokenizer.token_to_idx[tokenizer.UNKNOWN_TOKEN]:
            output_token = output_token[1]
        else:
            output_token = output_token[0]
        tokens.append(output_token)
        pad_mask.append(0)
        if output_token == tokenizer.token_to_idx[tokenizer.EOS_TOKEN]:
            break
    print(tokenizer.decode(tokens))

    # TODO: Peplexity


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
    parser.add_argument("--model-path", help="Path to saved model", required=True)

    args = parser.parse_args()
    main(args)
