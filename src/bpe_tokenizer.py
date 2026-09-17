from collections import Counter


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []  # список: (pair, new_token) — порядок важен!
        self.vocab = set()
    
    def _word_to_symbols(self, word):
        return list(word) + ['</w>']
    
    def _symbols_to_string(self, symbols):
        return ' '.join(symbols)
    
    def _string_to_symbols(self, s):
        return s.split()
    
    def _get_pair_frequencies(self, word_freqs):
        pairs = Counter()
        for word_str, freq in word_freqs.items():
            symbols = self._string_to_symbols(word_str)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_pair(self, word_freqs, pair):
        new_word_freqs = {}
        for word_str, freq in word_freqs.items():
            symbols = self._string_to_symbols(word_str)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                    new_symbols.append(symbols[i] + symbols[i+1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_freqs[self._symbols_to_string(new_symbols)] = freq
        return new_word_freqs
    
    def train(self, texts, num_merges=100):
        words = []
        for text in texts:
            words.extend(text.split())
        
        word_freqs = Counter(words)
        word_freqs = {
            self._symbols_to_string(self._word_to_symbols(word)): freq
            for word, freq in word_freqs.items()
        }
        
        self.vocab = set()
        for word_str in word_freqs:
            self.vocab.update(self._string_to_symbols(word_str))
        
        for merge_idx in range(num_merges):
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = best_pair[0] + best_pair[1]
            
            self.merges.append((best_pair, new_token))  # список, не словарь
            self.vocab.add(new_token)
            
            word_freqs = self._merge_pair(word_freqs, best_pair)
        
        self.token_to_id = {token: idx for idx, token in enumerate(sorted(self.vocab))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        return self
    
    def encode(self, text):
        """Текст -> список токенов."""
        words = text.split()
        token_ids = []
        
        for word in words:
            symbols = self._word_to_symbols(word)
            
            # Применяем merge'и в порядке обучения
            for pair, new_token in self.merges:
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            
            for sym in symbols:
                token_ids.append(self.token_to_id[sym])
        
        return token_ids
    
    def decode(self, token_ids):
        """Список токенов -> текст."""
        tokens = [self.id_to_token[idx] for idx in token_ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()
