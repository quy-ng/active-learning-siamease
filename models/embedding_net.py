import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.embeddings, self.embedding_dim = self._load_embeddings(embedding_dim)

    def _load_embeddings(self, embedding_dim):
        # todo: convert to static func
        word_embeddings = torch.nn.Embedding(embedding_dim[0], embedding_dim[1])
        emb_dim = embedding_dim[1]
        return word_embeddings, emb_dim

    def forward(self, x):
        output = self.embeddings(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def get_embedding_dim(self):
        return self.embedding_dim
