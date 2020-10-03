import torch
import torch.nn as nn
from ultils.character_level import generate_char_embedding, CHAR_EMBEDDING_INDEX
from torch.nn.utils.rnn import pad_sequence


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.embeddings = generate_char_embedding(CHAR_EMBEDDING_INDEX, embedding_dim)

    def forward(self, feed):
        X, X_len = self.load_padded_data(feed, CHAR_EMBEDDING_INDEX)
        return X, X_len

    def get_embedding(self, x):
        return self.forward(x)

    def get_embedding_dim(self):
        return self.embedding_dim

    def load_padded_data(self, feeds, word_to_index):
        """
        Padding data into a fixed length
        :return: x_train_pad: padded data
        :return: x_train_length: original length of all data (in case you want to unpadded)
        """
        x_train = []
        for index, row in enumerate(feeds):
            words_vector = [word_to_index.get(word) for word in row if word_to_index.get(word) is not None]
            x_train.append(torch.LongTensor(words_vector))

        x_train_len = [
            len(x) for x in x_train
        ]  # Get length for pack_padded_sequence after to remove padding
        x_train_pad = pad_sequence(x_train, batch_first=True)
        print("Load padded data successfully!")
        return x_train_pad, x_train_len


if __name__ == '__main__':
    from dataset import Inspectorio
    from dataset.augmentation import augment

    data = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment)
    print(data[1:3])

    embeddings_dim = 50
    batch_size = 1

    embed_net = EmbeddingNet(embeddings_dim)
    out = embed_net(data[1])
    print(out[0].shape)
    print('#'*10)
    out = embed_net(data[1:3])
    print(out[0].shape)
