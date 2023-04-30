import pickle
import re
from collections import defaultdict


class Tokenizer:
    def __init__(self, maxlen, minfreq, type, path=None):
        self.maxlen = maxlen
        self.minfreq = minfreq
        self.type = type
        # Special Tokens
        self.UNKNOWN_TOKEN = "<UNK>"
        self.EOS_TOKEN = "<EOS>"
        self.BOS_TOKEN = "<BOS>"
        self.PAD_TOKEN = "<PAD>"
        self.special_tokens = [
            self.UNKNOWN_TOKEN,
            self.EOS_TOKEN,
            self.BOS_TOKEN,
            self.PAD_TOKEN,
        ]

        self.vocab = list()  # List of unique words
        self.token_to_idx = dict()  # Mapping between word to index
        self.idx_to_token = dict()  # Mapping betwee index to word
        # Add special tokens to vocab
        self.vocab.extend(self.special_tokens)

        if path:
            self.load(path, self.type)

    def clean(self, sentence):
        # Lowercase
        sentence = sentence.lower()
        # Clean citation symbols like [1]
        sentence = re.sub(r"[\[0-9*\]]", "", sentence)
        # Convert [!, ?] to .
        sentence = sentence.replace("!", ".")
        sentence = sentence.replace("?", ".")
        # Remove punctuations except full stop
        sentence = re.sub(r"[^\w\s.]", "", sentence)
        # Add spaces to fullstop to treat it as a separate token
        sentence = re.sub("\.", " . ", sentence)
        return sentence

    def create_vocab(self, data):
        vocabulary = defaultdict(int)
        for sentence in data:
            sentence = self.clean(sentence)
            sequence = self.get_sequence(sentence)
            for t in sequence:
                vocabulary[t] += 1
        # Keep only high frequency tokens
        vocabulary = sorted(
            [token for token, freq in vocabulary.items() if freq >= self.minfreq]
        )
        self.vocab.extend(vocabulary)

        # Create index <-> token mapping
        for i, t in enumerate(self.vocab):
            self.token_to_idx[t] = i
            self.idx_to_token[i] = t

    def get_sequence(self, sentence):
        raise NotImplementedError

    # Map sentence to index sequence
    def encode(self, sentence, return_pad_mask=False, pad_to_max=False, add_eos=False):
        sentence = self.clean(sentence)
        tokens = [self.BOS_TOKEN]
        tokens.extend(
            self.get_sequence(sentence)[: self.maxlen - 2]
        )  # Makes sure the output is within maxlen
        # Add EOS
        if add_eos:
            tokens.append(self.EOS_TOKEN)
        # Add padding
        if pad_to_max:
            tokens.extend([self.PAD_TOKEN] * (self.maxlen - len(tokens)))
        # Convert to indices
        token_ids = [
            self.token_to_idx[w]
            if w in self.token_to_idx
            else self.token_to_idx[self.UNKNOWN_TOKEN]  # Replace OOV tokens with <UNK>
            for w in tokens
        ]
        if return_pad_mask:
            pad_mask = [
                0 if t == self.token_to_idx[self.PAD_TOKEN] else 1 for t in tokens
            ]
            return token_ids, pad_mask
        else:
            return token_ids

    # Map index sequence to sentence
    def decode(self, id_seq):
        tokens = [self.idx_to_token[i] for i in id_seq]
        tokens = [t for t in tokens if t != self.PAD_TOKEN]
        return " ".join(tokens)

    def save(self, fpath):
        save_dict = dict()
        save_dict["maxlen"] = self.maxlen
        save_dict["minfreq"] = self.minfreq
        save_dict["vocab"] = self.vocab
        save_dict["type"] = self.type
        with open(fpath, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, fpath, type):
        # Reinitialize to make sure everything is safe
        self.vocab = list()  # List of unique words
        self.token_to_idx = dict()  # Mapping between word to index
        self.idx_to_token = dict()  # Mapping betwee index to word

        with open(fpath, "rb") as f:
            saved_dict = pickle.load(f)

        if saved_dict["type"] != type:
            raise TypeError(
                f"Provided tokenizer path for {type} type but loading on {self.type} type"
            )
        if saved_dict["maxlen"] != self.maxlen or saved_dict["minfreq"] != self.minfreq:
            print("Overwriting tokenizer params using saved params")
        self.maxlen = saved_dict["maxlen"]
        self.minfreq = saved_dict["minfreq"]
        self.vocab = saved_dict["vocab"]

        # Create index <-> word mapping
        for i, w in enumerate(self.vocab):
            self.token_to_idx[w] = i
            self.idx_to_token[i] = w


class CharTokenizer(Tokenizer):
    def __init__(self, maxlen=256, minfreq=0, path=None):
        type = "char"
        super().__init__(maxlen, minfreq, type, path)

    def get_sequence(self, sentence):
        return list(sentence)


class WordTokenizer(Tokenizer):
    def __init__(self, maxlen=256, minfreq=0, path=None):
        type = "word"
        super().__init__(maxlen, minfreq, type, path)

    def get_sequence(self, sentence):
        return sentence.split()
