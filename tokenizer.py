import torch
from collections import defaultdict
import re
import json
from typing import List, Dict, Union

class Tokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.merges: Dict[tuple, str] = {}
        
        # Initialize special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def train(self, texts: List[str], min_freq: int = 2):
        # Initialize with characters
        word_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        # Preprocess texts
        for text in texts:
            # Split text into words
            words = text.lower().split()
            for word in words:
                # Add space before each word for better tokenization
                chars = ' ' + ' '.join(word)
                word_freqs[chars] += 1

        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs.keys():
            for char in word:
                vocab.add(char)

        # BPE training loop
        vocab_size = len(self.special_tokens) + len(vocab)
        
        while vocab_size < self.vocab_size:
            pair_freqs.clear()
            
            # Count pair frequencies
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Create new token
            new_token = ''.join(best_pair)
            vocab.add(new_token)
            self.merges[best_pair] = new_token

            # Update words with merged pairs
            new_word_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                new_word = word
                symbols = new_word.split()
                i = 0
                while i < len(symbols) - 1:
                    if (symbols[i], symbols[i + 1]) == best_pair:
                        symbols[i:i + 2] = [new_token]
                    i += 1
                new_word = ' '.join(symbols)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
            
            vocab_size += 1

        # Create final vocabulary
        for idx, token in enumerate(vocab, start=len(self.special_tokens)):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, text: str) -> torch.Tensor:
        tokens = ['<BOS>']
        
        # Tokenize word by word
        words = text.lower().split()
        for word in words:
            word = ' ' + ' '.join(word)
            while len(word.split()) > 1:
                changes = False
                for pair, merged in self.merges.items():
                    if f"{pair[0]} {pair[1]}" in word:
                        word = word.replace(f"{pair[0]} {pair[1]}", merged)
                        changes = True
                if not changes:
                    break
            tokens.extend(word.split())
        
        tokens.append('<EOS>')
        
        # Convert tokens to ids
        ids = [self.token_to_id.get(token, self.special_tokens['<UNK>']) 
               for token in tokens]
        
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        tokens = [self.id_to_token.get(id.item(), '<UNK>') for id in ids]
        # Remove special tokens and join
        text = ' '.join([token for token in tokens 
                        if token not in self.special_tokens.keys()])
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save(self, path: str):
        data = {
            'token_to_id': self.token_to_id,
            'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.merges = {tuple(k.split('|')): v for k, v in data['merges'].items()}

# Example usage
if __name__ == "__main__":
    # Example texts
    texts = [
        "Hello world!",
        "This is a test.",
        "Learning to tokenize text.",
    ]
    
    # Initialize and train tokenizer
    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.train(texts)
    
    # Test encoding and decoding
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Save and load test
    tokenizer.save("tokenizer.json")
    new_tokenizer = Tokenizer()
    new_tokenizer.load("tokenizer.json")