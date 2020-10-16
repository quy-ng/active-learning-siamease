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


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector:
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

    def get_triplets(self, embeddings, model, raw_data):

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
            hypothesis_positives = sorted_in[1:3].tolist()
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
            data_anchor = (raw_data[1][anchor][0], raw_data[1][anchor][1])
            data_test = (raw_data[1][test][0], raw_data[1][test][1])
            candidates.append((data_anchor, data_test, anchor, test))
        candidates.reverse()

        candidates = list(set(candidates))
        triplets = console_label(candidates)

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
