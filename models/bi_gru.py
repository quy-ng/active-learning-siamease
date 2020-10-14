import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class biGru(torch.nn.Module):
    """
    Triplet Model with embedding
    """

    def __init__(
            self,
            embedding_net,
            n_classes,
            hid_dim=120,
            layers=1,
            bidirectional=True
    ):
        super(biGru, self).__init__()
        self.n_classes = n_classes

        self.embed_layer = embedding_net
        self.is_cuda = embedding_net.is_cuda
        self.gru = torch.nn.GRU(
            self.embed_layer.get_embedding_dim(),
            hid_dim,
            layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3,
        )
        self.linear_final = torch.nn.Linear(2 * hid_dim, self.n_classes)  # turn output of gru to a vector

    def forward_detail(self, x):
        output, perm_idx = self.embed_layer(x)  # turn off when not use embedding
        output, _ = self.gru(output)
        output, output_lengths = pad_packed_sequence(
            output, batch_first=True
        )
        output = self.linear_final(output)
        return output, perm_idx

    def forward(self, x):
        embed_x, perm_idx = self.forward_detail(x)
        embed_x = torch.sum(embed_x, dim=1)
        return embed_x, perm_idx
