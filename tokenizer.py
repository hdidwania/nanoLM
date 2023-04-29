import pickle
import re
from collections import defaultdict


class Tokenizer:
    def __init__(self, maxlen=256, minfreq=0):
        self.maxlen = maxlen
        self.minfreq = minfreq

        self.vocab = list()  # List of unique words
        self.word_to_idx = dict()  # Mapping between word to index
        self.idx_to_word = dict()  # Mapping betwee index to word

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

        # Add special tokens to vocab
        self.vocab.extend(self.special_tokens)

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
            words = sentence.split()
            for w in words:
                vocabulary[w] += 1
        # Keep only high frequency words
        vocabulary = sorted(
            [word for word, freq in vocabulary.items() if freq >= self.minfreq]
        )
        self.vocab.extend(vocabulary)

        # Create index <-> word mapping
        for i, w in enumerate(self.vocab):
            self.word_to_idx[w] = i
            self.idx_to_word[i] = w

    # Map sentence to index sequence
    def encode(self, sentence, return_mask=False, pad_to_max=False, add_eos=False):
        sentence = self.clean(sentence)
        words = [self.BOS_TOKEN]
        words.extend(
            sentence.split()[: self.maxlen - 2]
        )  # Makes sure the output is within maxlen
        # Add EOS
        if add_eos:
            words.append(self.EOS_TOKEN)
        # Add padding
        if pad_to_max:
            words.extend([self.PAD_TOKEN] * (self.maxlen - len(words)))
        # Convert to indices
        tokens = [
            self.word_to_idx[w]
            if w in self.word_to_idx
            else self.word_to_idx[self.UNKNOWN_TOKEN]  # Replace OOV tokens with <UNK>
            for w in words
        ]
        if return_mask:
            mask = [0 if t == self.word_to_idx[self.PAD_TOKEN] else 1 for t in tokens]
            return tokens, mask
        else:
            return tokens

    # Map index sequence to sentence
    def decode(self, idx_seq):
        words = [self.idx_to_word[i] for i in idx_seq]
        words = [w for w in words if w != self.PAD_TOKEN]
        return " ".join(words)

    def save(self, fpath):
        save_dict = dict()
        save_dict["maxlen"] = self.maxlen
        save_dict["minfreq"] = self.minfreq
        save_dict["vocab"] = self.vocab
        with open(fpath, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, fpath):
        # Reinitialize to make sure everything is safe
        self.vocab = list()  # List of unique words
        self.word_to_idx = dict()  # Mapping between word to index
        self.idx_to_word = dict()  # Mapping betwee index to word

        with open(fpath, "rb") as f:
            saved_dict = pickle.load(f)

        if saved_dict["maxlen"] != self.maxlen or saved_dict["minfreq"] != self.minfreq:
            print("Overwriting tokenizer params using saved params")
        self.maxlen = saved_dict["maxlen"]
        self.minfreq = saved_dict["minfreq"]
        self.vocab = saved_dict["vocab"]

        # Create index <-> word mapping
        for i, w in enumerate(self.vocab):
            self.word_to_idx[w] = i
            self.idx_to_word[i] = w
