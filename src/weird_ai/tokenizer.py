class SimpleCharacterTokenizer:
    """
    Build a simple character tokenizer
    First, convert the text to a set to eliminate duplicates
    Next, convert the set to a list
    Sort the list, then assign it to the instance field chars
    Build a map of string to integer and the inverse, integer to string 
        This will convert strings to our tokens and back
    """
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        # String to integer
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        # Integer to string
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        """
        As an example, if the initialization of stoi mapped this:
        a = 0
        b = 1
        c = 2
        etc. and the itos simply swaps the keys/maps, then encoding the word cab
          will return [2, 0, 1]
        """
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens):
        """
        As an example, if the initialization of itos mapped this:
        0 = a
        1 = b
        2 = c
        etc. and the itos simply swaps the keys/maps, then decoding the list [2, 0, 1] 
        will return cab
        """
        return "".join(self.itos[token] for token in tokens)
    
