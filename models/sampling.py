from itertools import combinations
from typing import List, Tuple
import numpy as np
import torch
import copy
import random
from operator import itemgetter
from ultils.console_label import console_label


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + \
                      vectors.pow(2).sum(dim=1).view(1, -1) + \
                      vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """

    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, data, model):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, data, model):
        name_address = []
        for i in range(len(data[0])):
            name_address.append(data[0][i] + '; ' + data[1][i])
        embeddings, perm_idx = model(name_address)
        if self.cpu:
            embeddings = embeddings.cpu()

        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        triplets: List[Tuple] = []
        candidates: List[Tuple] = []
        candidates_index: List[Tuple] = []
        candidates_distance: List = []

        hypothesis_triplets = []

        for i in range(0, distance_matrix.shape[0]):
            copied_d = copy.copy(distance_matrix)
            sorted_val, sorted_in = copied_d[i].sort()
            hypothesis_positives = sorted_in[1:4].tolist()
            hypothesis_negatives = []

            # find hypothesis triplet where dp - dn < margin
            for j in hypothesis_positives:
                for k in range(0, distance_matrix.shape[0]):
                    if k not in hypothesis_positives and k != i:
                        hypothesis_negatives.append(k)
                _loss = distance_matrix[i, j] - distance_matrix[i, hypothesis_negatives] + self.margin
                _loss = _loss.data.cpu().numpy()
                hard_negative = np.argmin(_loss)
                hypothesis_triplets.append((i, j, hypothesis_negatives[hard_negative]))

            # mix hypothesis_triplets
            for l in hypothesis_triplets:
                candidates_index.append((l[0], l[1]))
                candidates_index.append((l[0], l[2]))
                candidates_index.append((l[1], l[2]))
            # mix hypothesis_positives
            candidates_index.append(tuple(random.sample(hypothesis_positives, 2)))

            # random
            random_selected = random.choices([i for i in range(distance_matrix.shape[0])], k=2)
            for h in random_selected:
                if i != h and (i, h) not in candidates_index:
                    candidates_index.append((i, h))

            candidates_index = list(set(candidates_index))
        for i in candidates_index:
            candidates_distance.append(distance_matrix[i[0], i[1]])

        indices, L_sorted = zip(*sorted(enumerate(candidates_distance), key=itemgetter(1)))
        indices_new = indices[:3] + indices[-3:] + indices[3:-3]
        for i in indices_new:
            anchor = candidates_index[i][0]
            test = candidates_index[i][1]
            data_anchor = (data[0][anchor], data[1][anchor])
            data_test = (data[0][test], data[1][test])
            candidates.append((data_anchor, data_test))
        candidates.reverse()

        candidates = list(set(candidates))
        triplets = console_label(candidates)
        anchor = []
        neg = []
        pos = []
        for i in triplets:
            anchor.append(i[0][0] + '; ' + i[0][1])
            pos.append(i[1][0] + '; ' + i[1][1])
            neg.append(i[2][0] + '; ' + i[2][1])
        anchor, _ = model(anchor)
        pos, _ = model(pos)
        neg, _ = model(neg)

        triplets = [anchor, pos, neg]
        triplets = np.array(triplets)

        return triplets


def HardestNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda
                                               x: semihard_negative(
                                               x, margin),
                                           cpu=cpu)
