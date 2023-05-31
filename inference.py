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


class TextGenerator:
    def __init__(
        self,
        tokenizer_type,
        tokenizer_path,
        tokenizer_maxlen,
        tokenizer_minfreq,
        model_dims,
        model_heads,
        model_blocks,
        model_path,
        sentence_maxlen,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer_class_dict[tokenizer_type](
            maxlen=tokenizer_maxlen, minfreq=tokenizer_minfreq, path=tokenizer_path
        )
        self.model = LanguageModel(
            dims=model_dims,
            heads=model_heads,
            nblocks=model_blocks,
            vocab_size=len(self.tokenizer.vocab),
            maxlen=MAXLEN,
            padding_idx=self.tokenizer.token_to_idx["<PAD>"],
            device=self.device,
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.sentence_maxlen = sentence_maxlen

        self.mode_to_fn = {
            "greedy": self.greedy_decode,
        }

    def generate(self, context, mode):
        tokens, pad_mask = self.tokenizer.encode(context, return_pad_mask=True)
        output = self.mode_to_fn[mode](tokens, pad_mask)
        return output

    def greedy_decode(self, tokens, pad_mask):
        while len(tokens) < self.sentence_maxlen:
            tokens_input = torch.tensor([tokens]).to(self.device)
            pad_mask_input = torch.tensor([pad_mask]).to(self.device)

            logits = self.model(tokens_input, pad_mask_input)
            logits = logits[0][-1]

            output_token = torch.topk(logits, k=2)
            output_token = output_token.indices.cpu().numpy().tolist()
            if (
                output_token[0]
                == self.tokenizer.token_to_idx[self.tokenizer.UNKNOWN_TOKEN]
            ):
                output_token = output_token[1]
            else:
                output_token = output_token[0]

            tokens.append(output_token)
            pad_mask.append(0)
            if output_token == self.tokenizer.token_to_idx[self.tokenizer.EOS_TOKEN]:
                break
        return self.tokenizer.decode(tokens)


def main(args):
    context_text = input("Enter Context: ")
    text_generator = TextGenerator(
        tokenizer_type=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        tokenizer_maxlen=args.maxlen,
        tokenizer_minfreq=args.minfreq,
        model_dims=args.dims,
        model_heads=args.heads,
        model_blocks=args.nblocks,
        model_path=args.model_path,
        sentence_maxlen=args.maxlen,
    )
    text = text_generator.generate(context_text, args.decode_mode)
    print(text)
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
    parser.add_argument("--model-path", help="Path to saved model", required=True)
    parser.add_argument("--decode-mode", help="Decoding strategy to use", required=True)

    args = parser.parse_args()
    main(args)
