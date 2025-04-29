import json
import time
from datetime import timedelta

class BPE:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = None
        self.merges = {}

    @staticmethod
    def get_stats(tokens):
        """Get the frequency of each pair of consecutive tokens."""
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids, pair, idx):
        """Merge the most frequent pair of tokens."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def fit(self, input_text, save_tokenizer=True):
        """Fit the BPE tokenizer to the input text."""
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string")
        byte_tokens = list(input_text.encode('utf-8'))
        self.merges = {}
        idx = 256
        total_merges = self.vocab_size - 256
        start_time = time.time()
        
        while idx < self.vocab_size:
            # Calculate progress
            progress = (idx - 256) / total_merges
            elapsed = time.time() - start_time
            
            if idx > 256: 
                eta_seconds = elapsed / (idx - 256) * (self.vocab_size - idx)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculating..."
            
            # Create and print progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f"\rProgress: [{bar}] {progress*100:.1f}% Complete ({idx-256}/{total_merges} merges) ETA: {eta_str}", end='')
            
            stats = self.get_stats(byte_tokens)
            if not stats:
                break              
            pair = max(stats, key=stats.get)
            self.merges[pair] = idx
            byte_tokens = self.merge(byte_tokens, pair, idx)
            idx += 1
        
        # Create vocab after fitting is complete
        self.make_vocab()
        
        if save_tokenizer:
            self.save("bpe_tokenizer.json")

    def make_vocab(self):
        """Create the vocabulary from the merges."""
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def decode(self, tokens):
        """Decode a sequence of byte tokens back to text."""
        if self.vocab is None:
            self.make_vocab()

        tokens = b"".join(self.vocab[idx] for idx in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        """Encode a string into byte tokens."""
        if not hasattr(self, 'merges') or not self.merges:
            raise ValueError("Tokenizer needs to be fitted with 'fit()' before encoding")
  
        byte_tokens = list(text.encode("utf-8"))
        while len(byte_tokens) >= 2:
            stats = self.get_stats(byte_tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            byte_tokens = self.merge(byte_tokens, pair, idx)
        return byte_tokens
    
    def save(self, filepath):
        """Save the tokenizer to a JSON file."""
        # Convert merges dict keys (tuples) to strings for JSON serialization
        serializable_merges = {}
        for (p0, p1), idx in self.merges.items():
            serializable_merges[f"{p0},{p1}"] = idx
            
        # Store vocab byte values as lists for JSON serialization
        serializable_vocab = {}
        if self.vocab:
            for idx, byte_val in self.vocab.items():
                serializable_vocab[str(idx)] = list(byte_val)
        
        data = {
            "vocab_size": self.vocab_size,
            "merges": serializable_merges,
            "vocab": serializable_vocab
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nTokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a tokenizer from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a new tokenizer instance
        tokenizer = cls(vocab_size=data["vocab_size"])
        
        # Restore merges
        tokenizer.merges = {}
        for pair_str, idx in data["merges"].items():
            p0, p1 = map(int, pair_str.split(','))
            tokenizer.merges[(p0, p1)] = idx
        
        # Restore vocab if it exists
        if "vocab" in data:
            tokenizer.vocab = {}
            for idx_str, byte_list in data["vocab"].items():
                tokenizer.vocab[int(idx_str)] = bytes(byte_list)
        
        return tokenizer


if __name__ == "__main__":
    # Test the BPE tokenizer with a sample dataset

    from datasets import load_dataset
    data = load_dataset('roneneldan/TinyStories')
    text = data['train']['text'][:1000]
    text = ''.join(text)
    
    print("Training tokenizer...")
    tokenizer = BPE(500)
    tokenizer.fit(text)
    tokenizer.save("bpe_tokenizer.json")
    
    # Test compression ratio
    raw = len(text.encode("utf-8"))
    merged = len(tokenizer.encode(text))
    print(f"Before: {raw}")
    print(f"After: {merged}")
    print(f"Compression ratio: {raw/merged:.2f}x")
    
    # Example of loading a saved tokenizer
    print("\nLoading saved tokenizer...")
    loaded_tokenizer = BPE.load("bpe_tokenizer.json")
    
    # Verify the loaded tokenizer works
    test_text = "Once upon a time there was a little girl."
    encoded = loaded_tokenizer.encode(test_text)
    decoded = loaded_tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test_text == decoded}")