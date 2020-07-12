import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletDistance(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        self.margin = margin
        super(TripletDistance, self).__init__()

    def forward(self, anchors, positives, negatives, size_average=True):
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
