import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
import string

from torch.autograd import Variable


class CharacterEmbedding(nn.Module):
    def __init__(self, embedding_size, max_length=None):
        super(CharacterEmbedding, self).__init__()
        self.vocab = ['<pad>'] + list(string.printable)
        self.embed = nn.Embedding(len(self.vocab), embedding_size)
        self.is_cuda = False
        self.cos = nn.CosineSimilarity(dim=2)
        self.embedding_size = embedding_size
        self.max_length = max_length

    def get_embedding_dim(self):
        return self.embedding_size

    def flatten(self, l):
        return list(itertools.chain.from_iterable(l))

    def embedAndPack(self, seqs, batch_first=False):

        vectorized_seqs = [[self.vocab.index(tok) for tok in seq] for seq in seqs]

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_lengths = seq_lengths.cuda() if self.is_cuda else seq_lengths

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        if self.max_length is None:
            self.max_length = seq_lengths.max()
        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), self.max_length))).long()
        seq_tensor = seq_tensor.cuda() if self.is_cuda else seq_tensor

        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda() if self.is_cuda else torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
        # Otherwise, give (L,B,D) tensors
        if not batch_first:
            seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)

        # embed your sequences
        seq_tensor = self.embed(seq_tensor)

        # pack them up nicely
        return pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True), perm_idx

    def cuda(self):
        self.is_cuda = True
        self.embed = self.embed.cuda()
        return self

    def forward(self, feed):
        output, perm_idx = self.embedAndPack(feed, batch_first=True)
        return output, perm_idx

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
    from dataset import Inspectorio
    from dataset.augmentation import augment_dataframe

    data = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment_dataframe)
    batch_data = data[1:10]

    embeddings_dim = 50

    embed_net = CharacterEmbedding(embeddings_dim)
    out, sorted_idx = embed_net(batch_data)
    print(out)
    print('#' * 10)

    words = embed_net.unpackToSequence(out)
    print(words)
    print([batch_data[i] for i in sorted_idx])

    print('#' * 10)
    n_classes = 10
    emb_dim = embed_net.get_embedding_dim()
    hid_dim = 50
    layers = 1
    bidirectional = True
    gru = torch.nn.GRU(
        emb_dim,
        hid_dim,
        layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=0.3,
    )
    linear_final = torch.nn.Linear(2 * hid_dim, n_classes)  # turn output of gru to a vector
    x_packed, _ = embed_net(batch_data)
    x_packed, hidden_state = gru(x_packed)
    output, output_lengths = pad_packed_sequence(
        x_packed, batch_first=True
    )
    final_out = linear_final(output)
    print(final_out)
