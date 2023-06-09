import argparse

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import tokenizer as tokenizer_
from dataset import MovieData
from model import LanguageModel

tokenizer_class_dict = {
    "word": tokenizer_.WordTokenizer,
    "char": tokenizer_.CharTokenizer,
}

SAVE_PATH = "saved-models/model-checkpoint-{}.pt"
LOSS_PLOT_PATH = "saved-models/train-loss-curve.jpg"
PERPLEXITY_PLOT_PATH = "saved-models/perplexity-curve.jpg"


def forward_pass_for_batch(
    sentence_batch, model, tokenizer, loss_fn, device, mode="train"
):
    sentence_batch = [
        tokenizer.encode(sentence, return_pad_mask=True, pad_to_max=True, add_eos=True)
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

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    output_logits = model(encoded_input, sentence_pad_masks)
    loss_value = loss_fn(output_logits.transpose(-1, -2), encoded_output)
    loss_value = loss_value.mean()
    return loss_value


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preparing Dataset")
    train_data = MovieData(maxlen=args.maxlen, split="train")
    val_data = MovieData(maxlen=args.maxlen, split="val")
    print("Number of train and val samples:", len(train_data), len(val_data))
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

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    loss_fn = CrossEntropyLoss(reduction="none").to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    i_step_global = 0
    loss_values_list = list()
    train_perplexity_list = list()
    val_perplexity_list = list()
    for i_epoch in range(args.n_epochs):
        epoch_loss_sum = 0
        for i_step, sentence_batch in enumerate(train_dataloader):
            i_step_global += 1
            optimizer.zero_grad()
            loss_value = forward_pass_for_batch(
                sentence_batch, model, tokenizer, loss_fn, device, mode="train"
            )
            loss_value.backward()
            optimizer.step()
            epoch_loss_sum += loss_value
            print(
                "\rGlobal Step {:3d} | Epoch {:2d}/{:2d} Step {:3d}/{:3d} | LR {:e} | Loss {:.4f}".format(
                    i_step_global,
                    i_epoch + 1,
                    args.n_epochs,
                    i_step + 1,
                    len(train_dataloader),
                    lr_scheduler.get_last_lr()[-1],
                    epoch_loss_sum / (i_step + 1),
                ),
                end="",
            )

            # Save model
            if i_step_global % args.save_after_step == 0:
                torch.save(model.state_dict(), SAVE_PATH.format(i_step_global))
        print()
        lr_scheduler.step()
        loss_values_list.append(epoch_loss_sum.item() / (i_step + 1))

        # Train set perplexity:
        with torch.no_grad():
            train_loss_sum = 0.0
            for i_step, sentence_batch in enumerate(train_dataloader):
                loss_value = forward_pass_for_batch(
                    sentence_batch, model, tokenizer, loss_fn, device, mode="val"
                )
                train_loss_sum += loss_value
                print(
                    "\rCalculating Train Perplexity Step {}/{}".format(
                        i_step + 1, len(train_dataloader)
                    ),
                    end="",
                )
            train_perplexity_list.append(
                torch.exp(train_loss_sum / len(train_dataloader)).item()
            )
            print()

            val_loss_sum = 0.0
            for i_step, sentence_batch in enumerate(val_dataloader):
                loss_value = forward_pass_for_batch(
                    sentence_batch, model, tokenizer, loss_fn, device, mode="val"
                )
                val_loss_sum += loss_value
                print(
                    "\rCalculating Val Perplexity Step {}/{}".format(
                        i_step + 1, len(val_dataloader)
                    ),
                    end="",
                )
            val_perplexity_list.append(
                torch.exp(val_loss_sum / len(val_dataloader)).item()
            )
            print()

    plt.plot(range(1, args.n_epochs + 1), loss_values_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    plt.plot(range(1, args.n_epochs + 1), train_perplexity_list)
    plt.plot(range(1, args.n_epochs + 1), val_perplexity_list)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.legend(["Train Data", "Val Data"])
    plt.savefig(PERPLEXITY_PLOT_PATH)
    plt.close()


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
        "--tokenizer-path",
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
