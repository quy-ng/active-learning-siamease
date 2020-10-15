import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ultils.character_level import vectorize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
__all__ = ['OnlineTripletLoss', 'TripletDistance']


def create_data_loader(X, batch_size):
    X, X_lens = np.array(X[0]), np.array(X[1])

    # Create data loader
    data = TensorDataset(
        torch.from_numpy(X).type(torch.LongTensor), torch.ByteTensor(X_lens)
    )
    loader = DataLoader(data, batch_size=batch_size, drop_last=False)
    return loader


class TripletDistance(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        self.margin = margin
        super(TripletDistance, self).__init__()

    def forward(self, anchors, positives, negatives):
        anchors_2d = anchors.reshape(anchors.shape[0], -1)
        positives_2d = positives.reshape(positives.shape[0], -1)
        negatives_2d = negatives.reshape(negatives.shape[0], -1)

        similarity_pos = torch.sum(anchors_2d * positives_2d, dim=1) / (
                torch.sqrt(torch.sum(anchors_2d * anchors_2d, dim=1))
                * torch.sqrt(torch.sum(positives_2d * positives_2d, dim=1))
        )

        similarity_neg = torch.sum(anchors_2d * negatives_2d, dim=1) / (
                torch.sqrt(torch.sum(anchors_2d * anchors_2d, dim=1))
                * torch.sqrt(torch.sum(negatives_2d * negatives_2d, dim=1))
        )

        losses = F.relu(-similarity_pos + similarity_neg + self.margin)

        return (
            losses.sum(),
            similarity_pos,
            similarity_neg,
        )


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector, triplet_distance):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.triplet_distance = triplet_distance

    def forward(self, embedded_data, model, raw_data, vocab):

        triplets = self.triplet_selector.get_triplets(embedded_data, model, raw_data)

        if model.is_cuda:
            triplets = triplets.cuda()

        a = []
        n = []
        p = []
        for i in triplets:
            a.append(i[0][1])
            p.append(i[1][1])
            n.append(i[2][1])

        aa = []
        aa_length = []
        for i in a:
            chars_vector = vectorize(i, vocab)
            aa.append(torch.LongTensor(chars_vector))
            aa_length.append(len(i))

        nn = []
        nn_length = []
        for i in n:
            chars_vector = vectorize(i, vocab)
            nn.append(torch.LongTensor(chars_vector))
            nn_length.append(len(i))

        pp = []
        pp_length = []
        for i in p:
            chars_vector = vectorize(i, vocab)
            pp.append(torch.LongTensor(chars_vector))
            pp_length.append(len(i))

        aa_train = pad_sequence(aa, batch_first=True)
        nn_train = pad_sequence(nn, batch_first=True)
        pp_train = pad_sequence(pp, batch_first=True)
        aa_length = torch.ByteTensor(aa_length)
        nn_length = torch.ByteTensor(nn_length)
        pp_length = torch.ByteTensor(pp_length)

        anchor = create_data_loader([aa_train, aa_length], len(aa))
        positive = create_data_loader([pp_train, pp_length], len(pp))
        negative = create_data_loader([nn_train, nn_length], len(nn))

        x, pos, neg = model(anchor, positive, negative)

        loss, pos_sim, neg_sim = self.triplet_distance(x, pos, neg)

        return loss, pos_sim, neg_sim
