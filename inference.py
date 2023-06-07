import argparse

import numpy as np
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
            "beam-search": self.beam_search,
            "sample": self.sample,
        }

    def generate(self, context, mode, num_beams, temperature):
        tokens, pad_mask = self.tokenizer.encode(context, return_pad_mask=True)
        output = self.mode_to_fn[mode](
            tokens=tokens,
            pad_mask=pad_mask,
            num_beams=num_beams,
            temperature=temperature,
        )
        return output

    def get_logits(self, tokens, mask):
        token_tensor = torch.tensor([tokens]).to(self.device)
        mask_tensor = torch.tensor([mask]).to(self.device)

        logits = self.model(token_tensor, mask_tensor)
        logits = logits[0][-1]
        return logits.detach()

    def greedy_decode(self, **kwargs):
        tokens = kwargs.get("tokens")
        pad_mask = kwargs.get("pad_mask")

        while len(tokens) < self.sentence_maxlen:
            logits = self.get_logits(tokens, pad_mask)

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

    def beam_search(self, **kwargs):
        tokens = kwargs.get("tokens")
        pad_mask = kwargs.get("pad_mask")
        num_beams = kwargs.get("num_beams")
        assert num_beams != None

        hypotheses = list()
        hypotheses.append([tokens, pad_mask, 0.0])
        while True:
            new_hypotheses = list()
            for h_tokens, h_mask, h_score in hypotheses:
                if (
                    len(h_tokens) >= self.sentence_maxlen
                    or h_tokens[-1]
                    == self.tokenizer.token_to_idx[self.tokenizer.EOS_TOKEN]
                ):
                    continue
                logits = self.get_logits(h_tokens, h_mask)
                output_tokens = torch.topk(logits, k=num_beams + 1)
                output_scores = output_tokens.values.cpu().numpy().tolist()
                output_tokens = output_tokens.indices.cpu().numpy().tolist()
                for i in range(num_beams + 1):
                    if (
                        output_tokens[i]
                        == self.tokenizer.token_to_idx[self.tokenizer.UNKNOWN_TOKEN]
                    ):
                        continue
                    new_hypotheses.append(
                        [
                            h_tokens + [output_tokens[i]],
                            h_mask + [0],
                            h_score + np.log(output_scores[i]),
                        ]
                    )
            hypotheses = hypotheses + new_hypotheses
            hypotheses.sort(key=lambda x: x[-1], reverse=True)
            hypotheses = hypotheses[:num_beams]
            if len(new_hypotheses) == 0:
                break
        return self.tokenizer.decode(hypotheses[0][0])

    def sample(self, **kwargs):
        tokens = kwargs.get("tokens")
        pad_mask = kwargs.get("pad_mask")
        temperature = kwargs.get("temperature")
        assert temperature != None

        while len(tokens) < self.sentence_maxlen:
            logits = self.get_logits(tokens, pad_mask)
            token_probs = torch.softmax(logits / temperature, dim=0).cpu().numpy()
            output_token = self.tokenizer.token_to_idx[self.tokenizer.UNKNOWN_TOKEN]
            while (
                output_token
                == self.tokenizer.token_to_idx[self.tokenizer.UNKNOWN_TOKEN]
            ):
                output_token = np.random.choice(
                    range(0, len(token_probs)), p=token_probs
                )
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
    text = text_generator.generate(
        context_text, args.decode_mode, args.num_beams, args.temperature
    )
    print(text)


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
    parser.add_argument("--num-beams", help="Beam Width for Beam Search", type=int)
    parser.add_argument("--temperature", help="Temperature for sampling", type=float)

    args = parser.parse_args()
    main(args)
