import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
import numpy as np
from ultils.character_level import default_vocab


class CharacterEmbedding(nn.Module):
    def __init__(self, embedding_size, vocab=None, max_length=None):
        super(CharacterEmbedding, self).__init__()
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = default_vocab

        self.embed = nn.Embedding(len(self.vocab), embedding_size)
        self.is_cuda = False
        self.cos = nn.CosineSimilarity(dim=2)
        self.embedding_size = embedding_size
        self.max_length = max_length

    def get_embedding_dim(self):
        return self.embedding_size

    def flatten(self, l):
        return list(itertools.chain.from_iterable(l))

    def embedAndPack(self, vectorized_seqs, seq_lengths=None, batch_first=True, enforce_sorted=False):

        if seq_lengths is None:
            seq_lengths = [len(x) for x in vectorized_seqs]

        # # dump padding everywhere, and place seqs on the left.
        # # NOTE: you only need a tensor as big as your longest sequence
        # seq_tensor = np.zeros((len(vectorized_seqs), self.max_length))
        # for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        #     seq_tensor[idx, :seqlen] = seq

        # embed your sequences
        # seq_tensor = torch.from_numpy(seq_tensor).type(torch.LongTensor)
        # seq_tensor = self.embed(seq_tensor)
        seq_tensor = self.embed(vectorized_seqs)

        # pack them up nicely
        return pack_padded_sequence(seq_tensor, seq_lengths,
                                    batch_first=batch_first,
                                    enforce_sorted=enforce_sorted)

    def cuda(self):
        self.is_cuda = True
        self.embed = self.embed.cuda()
        return self

    def forward(self, feed):
        if type(feed) is list:
            output = self.embedAndPack(feed[0], feed[1], batch_first=True)
        return output

    def unpackToSequence(self, packed_output):
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        words = self.unembed(output)
        return words

    def unembed(self, embedded_sequence):
        weights = self.embed.state_dict()['weight']
        weights = weights.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        e_sequence = embedded_sequence.unsqueeze(3).data
        cosines = self.cos(e_sequence, weights)
        _, indexes = torch.topk(cosines, 1, dim=2)

        words = []
        for word in indexes:
            word_l = ''
            for char_index in word:
                word_l += self.vocab[char_index[0]]
            words.append(word_l)
        return words


if __name__ == '__main__':
    from ultils.character_level import vectorize
    embeddings_dim = 50
    length_max = 60
    batch_data = [
        'no. 165 wehou ave, economic & technology deelpmet zone',
        '90 udog vihar, phas i'
    ]

    embed_net = CharacterEmbedding(embedding_size=embeddings_dim,
                                   max_length=length_max)
    vectorized_seqs, data_len = vectorize(batch_data, embed_net.vocab)
    out = embed_net(vectorized_seqs)
    print(out)
    print('#' * 10)

    words = embed_net.unpackToSequence(out)
    print(words)

