import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models import EmbeddingNet


class TripletNet(torch.nn.Module):
    """
    Triplet Model with embedding
    """

    def __init__(
            self,
            embedding_dim,  # [59, 50]
            n_classes,
            hid_dim=120,
            layers=1,
            bidirectional=True
    ):
        super(TripletNet, self).__init__()
        self.n_classes = n_classes

        self.embed_layer = EmbeddingNet(embedding_dim)
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
        x, x_len = x
        x_embed = self.embeddings(x)  # turn off when not use embedding
        x_packed = pack_padded_sequence(
            x_embed, x_len, batch_first=True, enforce_sorted=False
        )
        x_packed, _ = self.gru(x_packed)
        output, output_lengths = pad_packed_sequence(
            x_packed, batch_first=True
        )
        return self.linear_final(output)

    def forward(self, x1, x2, x3=None):
        anchors, positives, negatives = (
            self.forward_detail(x1),
            self.forward_detail(x2),
            self.forward_detail(x3),
        )
        # make 2 vectors from bi-LSTM to one
        anchors, positives, negatives = (
            torch.sum(anchors, dim=1),
            torch.sum(positives, dim=1),
            torch.sum(negatives, dim=1),
        )
        return anchors, positives, negatives
