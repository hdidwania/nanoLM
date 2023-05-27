import argparse
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import tokenizer as tokenizer_
from dataset import MovieData
from model import LanguageModel

######### TODO MAIN #################
# SELECT ANOTHER DATASET
# OR CLEAN NON ENGLISH DATA
####################################

tokenizer_class_dict = {
    "word": tokenizer_.WordTokenizer,
    "char": tokenizer_.CharTokenizer,
}
SAVE_PATH = "saved-models/model-checkpoint-{}.pt"


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preparing Dataset")
    data = MovieData(maxlen=args.maxlen)
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
        maxlen=args.maxlen,
        padding_idx=tokenizer.token_to_idx["<PAD>"],
        device=device,
    )

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    loss_fn = CrossEntropyLoss(reduction="none").to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    i_step_global = 0
    for i_epoch in range(args.n_epochs):
        epoch_loss_sum = 0
        for i_step, sentence_batch in enumerate(dataloader):
            i_step_global += 1
            sentence_batch = [
                tokenizer.encode(
                    sentence, return_pad_mask=True, pad_to_max=True, add_eos=True
                )
                for sentence in sentence_batch
            ]
            sentence_token_ids, sentence_pad_masks = list(zip(*sentence_batch))
            sentence_token_ids = torch.tensor(sentence_token_ids).to(device)
            # For a sequence 1..N, use 1..N-1 as input and 2..N as output
            encoded_input = sentence_token_ids[:, :-1]
            encoded_input.requires_grad = False
            encoded_output = sentence_token_ids[:, 1:]
            encoded_output.requires_grad = False
            sentence_pad_masks = torch.tensor(sentence_pad_masks).to(device)
            sentence_pad_masks = sentence_pad_masks[:, :-1]

            model.train()
            optimizer.zero_grad()
            output_logits = model(encoded_input, sentence_pad_masks)
            loss_val = loss_fn(output_logits.transpose(-1, -2), encoded_output)
            loss_val = loss_val.mean()
            loss_val.backward()
            optimizer.step()

            epoch_loss_sum += loss_val
            print(
                "\rGlobal Step {:3d} | Epoch {:2d}/{:2d} Step {:3d}/{:3d} | Loss {:.4f}".format(
                    i_step_global,
                    i_epoch + 1,
                    args.n_epochs,
                    i_step + 1,
                    len(dataloader),
                    epoch_loss_sum / (i_step + 1),
                ),
                end="",
            )

            # Save model
            if i_step_global % args.save_after_step == 0:
                torch.save(model.state_dict(), SAVE_PATH.format(i_step_global))

        print()
    # TODO Train test split?
    # TODO End of epoch test on test data?
    # TODO Number of params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--nblocks", type=int, default=3)
    parser.add_argument("--save_after_step", type=int, default=500)
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
