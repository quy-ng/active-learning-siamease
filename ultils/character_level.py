# import string
__all__ = ['default_vocab', 'vectorize']

# char_targets = string.printable[:36] + \
#                        string.printable[63:65] + \
#                        string.printable[67:71] + \
#                        string.printable[74:78] + ',;@' + ' ' + "，"
# default_vocab = ['<pad>'] + list(char_targets)

default_vocab = {
    "<PAD>": 0,
    "，": 1,
    " ": 2,
    "#": 3,
    "&": 4,
    "'": 5,
    "(": 6,
    ")": 7,
    ",": 8,
    "-": 9,
    ".": 10,
    "/": 11,
    ":": 12,
    ";": 13,
    "0": 14,
    "1": 15,
    "2": 16,
    "3": 17,
    "4": 18,
    "5": 19,
    "6": 20,
    "7": 21,
    "8": 22,
    "9": 23,
    "a": 24,
    "b": 25,
    "c": 26,
    "d": 27,
    "e": 28,
    "f": 29,
    "g": 30,
    "h": 31,
    "i": 32,
    "j": 33,
    "k": 34,
    "l": 35,
    "m": 36,
    "n": 37,
    "o": 38,
    "p": 39,
    "q": 40,
    "r": 41,
    "s": 42,
    "t": 43,
    "u": 44,
    "v": 45,
    "w": 46,
    "x": 47,
    "y": 48,
    "z": 49
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
