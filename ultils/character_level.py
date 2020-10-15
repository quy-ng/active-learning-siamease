import string
__all__ = ['default_vocab', 'vectorize']

char_targets = string.printable[:36] + \
                       string.printable[63:65] + \
                       string.printable[67:71] + \
                       string.printable[74:78] + ',;@' + ' ' + '，'
# default_vocab = ['<pad>'] + list(char_targets)

default_vocab = {
    " ": 1,
    "!": 2,
    "#": 3,
    "$": 4,
    "%": 5,
    "&": 6,
    "'": 7,
    "(": 8,
    ")": 9,
    "*": 10,
    ",": 11,
    "-": 12,
    ".": 13,
    "/": 14,
    "0": 15,
    "1": 16,
    "2": 17,
    "3": 18,
    "4": 19,
    "5": 20,
    "6": 21,
    "7": 22,
    "8": 23,
    "9": 24,
    ":": 25,
    ";": 26,
    "<": 27,
    ">": 28,
    "?": 29,
    "@": 30,
    "^": 31,
    "a": 32,
    "b": 33,
    "c": 34,
    "d": 35,
    "e": 36,
    "f": 37,
    "g": 38,
    "h": 39,
    "i": 40,
    "j": 41,
    "k": 42,
    "l": 43,
    "m": 44,
    "n": 45,
    "o": 46,
    "p": 47,
    "q": 48,
    "r": 49,
    "s": 50,
    "t": 51,
    "u": 52,
    "v": 53,
    "w": 54,
    "x": 55,
    "y": 56,
    "z": 57,
    "，": 58,
    "<PAD>": 0,
}


# def vectorize(seqs, vocab):
#     vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
#     data_len = [
#         len(x) for x in vectorized_seqs
#     ]  # Get length for pack_padded_sequence after to remove padding
#     return vectorized_seqs, data_len

def vectorize(seq, vocab):
    try:
        vectorized_seq = [vocab.get(tok) for tok in seq if vocab.get(tok) is not None]
    except:
        print(seq)
    return vectorized_seq
