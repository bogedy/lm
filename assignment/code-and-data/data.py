from __future__ import annotations
from typing import Iterator
import torch
from torch import nn
import torch.nn.functional as F
import random
import glob
import json

class CharTokenizer:
    def __init__(self):
        self.symbols = ["<PAD>"]
        self.tokens = set()
        self.vocab = list(self.symbols)
        self.stoi = {s:i for i, s in enumerate(self.vocab)}

    def pad_id(self): return self.stoi["<PAD>"]

    def get_id(self, tok: str): return self.stoi[tok]

    def vocab_size(self): return len(self.vocab)
        
    def train(self, sequences: list[str]) -> None:
        for seq in sequences:
            for symbol in self._tokenize_to_symbols(seq):
                self.tokens.add(symbol)

        self.vocab = list(self.symbols) + list(sorted(self.tokens))
        self.stoi = {s:i for i, s in enumerate(self.vocab)}


    def _tokenize_to_symbols(self, text: str) -> list[str]:
        return list(text)

    def tokenize(self, text: str) -> list[int]:
        seq: list[str] = self._tokenize_to_symbols(text)
        return [self.stoi[s] for s in seq]

    def detokenize(self, tokens: list[int], keep_symbols = True) -> str:
        strs: list[str] = [self.vocab[t] for t in tokens]
        if not keep_symbols:
            strs = [s for s in strs if len(s) == 1]
        return "".join(strs)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "symbols": self.symbols,
                "tokens": list(self.tokens),
                "vocab": self.vocab,
                "stoi": self.stoi
            }, f)

    @staticmethod
    def load(path: str) -> CharTokenizer:
        tokenizer = CharTokenizer()
        with open(path, "r") as f:
            data = json.load(f)
            tokenizer.symbols = data["symbols"]
            tokenizer.tokens = set(data["tokens"])
            tokenizer.vocab = data["vocab"]
            tokenizer.stoi = data["stoi"]
        return tokenizer

class RandomOrderDataIterator:
    def __init__(self, data, desired_length):
        self.desired_length = desired_length
        self.data: list[list[int]] = [seq for seq in data if len(seq) > self.desired_length]

    def __iter__(self):
        if len(self.data) == 0: return
        while True:
            seq = random.choice(self.data)
            idx = random.randint(0, len(seq) - self.desired_length)
            yield seq[idx:idx + self.desired_length]


# This both creates the tokenizer and uses it to tokenize the data.
# In a real system you'd like to split it to two separate functions.
# Feel free to separate it to two functions also in this code.
def load_data(path: str) -> tuple[CharTokenizer, list[list[int]], list[list[int]]]:
    tokenizer = CharTokenizer()
    for fname in glob.glob(f"{path}/*.txt"):
        with open(fname) as fh:
            text = fh.read()
            tokenizer.train(text)

    data: list[list[int]] = []
    for fname in glob.glob(f"{path}/*.txt"):
        with open(fname) as fh:
            text = fh.read()
            data.append(tokenizer.tokenize(text))

    test_split = 0.1
    if len(data)==1:
        split_idx = int(len(data[0]) * (1 - test_split))
        train_data = [data[0][:split_idx]]
        test_data = [data[0][split_idx:]]
        print("train test lengths:", len(train_data[0]), len(test_data[0]))
    else:
        split_idx = int(len(data) * (1 - test_split))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        print("train test lengths:", len(train_data[0]), print(len(test_data[0])))

    return (tokenizer, train_data, test_data)

def batch_items(data_iter: Iterator[list[int]], batch_size: int = 2) -> Iterator[torch.LongTensor]:
    batch = []
    for seq in data_iter:
        idx = 0
        batch.append(seq)
        if len(batch) >= batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
    if len(batch) > 0:
        yield torch.tensor(batch, dtype=torch.long)

