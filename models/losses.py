import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ultils.character_level import vectorize
from torch.nn.utils.rnn import pad_sequence
__all__ = ['OnlineTripletLoss', 'TripletDistance']


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

    def __init__(self, margin, triplet_selector, triplet_distance, max_length, is_web=False, task_id=None):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.triplet_distance = triplet_distance
        self.max_length = max_length
        self.is_web = is_web
        self.task_id = task_id

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
            aa.append(chars_vector)
            aa_length.append(len(chars_vector))

        nn = []
        nn_length = []
        for i in n:
            chars_vector = vectorize(i, vocab)
            nn.append(chars_vector)
            nn_length.append(len(chars_vector))

        pp = []
        pp_length = []
        for i in p:
            chars_vector = vectorize(i, vocab)
            pp.append(chars_vector)
            pp_length.append(len(chars_vector))

        aa_tensor = np.zeros((len(aa), self.max_length))
        for idx, (seq, seqlen) in enumerate(zip(aa, aa_length)):
            aa_tensor[idx, :seqlen] = seq
        aa_tensor = torch.from_numpy(aa_tensor).type(torch.LongTensor)

        nn_tensor = np.zeros((len(nn), self.max_length))
        for idx, (seq, seqlen) in enumerate(zip(nn, nn_length)):
            nn_tensor[idx, :seqlen] = seq
        nn_tensor = torch.from_numpy(nn_tensor).type(torch.LongTensor)

        pp_tensor = np.zeros((len(pp), self.max_length))
        for idx, (seq, seqlen) in enumerate(zip(pp, pp_length)):
            pp_tensor[idx, :seqlen] = seq
        pp_tensor = torch.from_numpy(pp_tensor).type(torch.LongTensor)

        aa_train = pad_sequence(aa_tensor, batch_first=True)
        nn_train = pad_sequence(nn_tensor, batch_first=True)
        pp_train = pad_sequence(pp_tensor, batch_first=True)
        aa_length = torch.ByteTensor(aa_length)
        nn_length = torch.ByteTensor(nn_length)
        pp_length = torch.ByteTensor(pp_length)

        anchor = [aa_train, aa_length]
        positive = [pp_train, pp_length]
        negative = [nn_train, nn_length]

        x, pos, neg = model(anchor, positive, negative)

        loss, pos_sim, neg_sim = self.triplet_distance(x, pos, neg)

        return loss, pos_sim, neg_sim, len(triplets)
