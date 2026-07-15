"""Custom Byte-Pair Encoding (BPE) Tokenizer.
Trains on the provided corpus to compress Hindi (Devanagari) characters efficiently
while maintaining a strict lossless UTF-8 byte fallback.
"""
import json
import os

class BPETokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size=512):
        # Speed hack: Train on the first 500k bytes so we don't waste our hour
        tokens = list(text[:500000].encode("utf-8"))
        num_merges = vocab_size - 256
        
        print(f"Training BPE to {vocab_size} tokens... this will take about 15 seconds.")
        for i in range(num_merges):
            counts = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j+1])
                counts[pair] = counts.get(pair, 0) + 1
            
            if not counts:
                break
            
            best_pair = max(counts, key=counts.get)
            new_token_id = 256 + i
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(new_token_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            
        self.vocab_size = 256 + len(self.merges)
        print(f"BPE training complete. New Vocab size: {self.vocab_size}")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    stats[i] = self.merges[pair]
            
            if not stats:
                break
            
            best_idx = min(stats, key=lambda k: stats[k])
            best_pair = (tokens[best_idx], tokens[best_idx+1])
            new_token_id = self.merges[best_pair]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        b = bytearray()
        for idx in ids:
            b.extend(self.vocab[idx])
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        m = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w") as f:
            json.dump({"type": "bpe", "vocab_size": self.vocab_size, "merges": m}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)} # Initialize base bytes
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(","))
            self.merges[(p1, p2)] = v
            self.vocab[v] = self.vocab[p1] + self.vocab[p2]

def load(path="tokenizer.json"):
    """Return the tokenizer used by evaluate.py. Replace as needed."""
    tok = BPETokenizer()
    if os.path.exists(path):
        tok.load(path)
    else:
        print("No saved tokenizer found. Initializing auto-training...")
        text = open("../data/train_corpus.txt", encoding="utf-8").read()
        tok.train(text, vocab_size=512)
        tok.save(path)
    return tok