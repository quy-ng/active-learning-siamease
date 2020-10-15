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
        output = self.embed_layer(x)  # turn off when not use embedding
        output, _ = self.gru(output)
        output, output_lengths = pad_packed_sequence(
            output, batch_first=True
        )
        output = self.linear_final(output)
        return output

    def forward(self, x1, x2=None, x3=None):
        """
        x1, x2, x3: List[Tensor], List[0] is data, List[1] data length
        """
        if x3 is not None:
            anchors, positives, negatives = (
                self.forward_detail(x1),
                self.forward_detail(x2),
                self.forward_detail(x3),
            )
            anchors, positives, negatives = (
                torch.sum(anchors, dim=1),
                torch.sum(positives, dim=1),
                torch.sum(negatives, dim=1),
            )
            return anchors, positives, negatives
        elif x2 is None and x3 is None:
            # for active learning
            x1 = self.forward_detail(x1)
            x1 = torch.sum(x1, dim=1)
            return x1
        else:
            # for test
            x1, x2 = self.forward_detail(x1), self.forward_detail(x2)
            x1, x2 = torch.sum(x1, dim=1), torch.sum(x2, dim=1)
            return x1, x2
