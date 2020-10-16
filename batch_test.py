import warnings
import torch
import numpy as np
import itertools
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
import pandas as pd
from models import CharacterEmbedding, biGru
from models.losses import TripletDistance
from ultils.character_level import default_vocab
from dataset import load_padded_data
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


def data_loader(test_df_1, test_df_2, embedding_index, batch_size):
    """
    Generate dataloader to test the result
    @param test_df_1:
    @param test_df_2:
    @param embedding_index:
    @param batch_size:
    @return: two dataloaders and removed indexes
    """
    # Data Preparation pipeline
    # Create Dataloader based on two dataframe have row 'content' in it
    X1, X1_lens = load_padded_data(pd.DataFrame(test_df_1), embedding_index)
    X2, X2_lens = load_padded_data(pd.DataFrame(test_df_2), embedding_index)

    # Drop rows that have length of word vector = 0
    truncate_index = [
        i for i in range(0, len(X1_lens)) if (X1_lens[i] <= 0 or X2_lens[i] <= 0)
    ]
    X1, X1_lens = (
        np.delete(X1, truncate_index, axis=0),
        np.delete(X1_lens, truncate_index, axis=0),
    )
    X2, X2_lens = (
        np.delete(X2, truncate_index, axis=0),
        np.delete(X2_lens, truncate_index, axis=0),
    )

    def create_data_loader(X, batch_size=batch_size):
        X, X_lens = np.array(X[0]), np.array(X[1])

        # Create data loader
        data = TensorDataset(
            torch.from_numpy(X).type(torch.LongTensor), torch.ByteTensor(X_lens)
        )
        loader = DataLoader(data, batch_size=batch_size, drop_last=False)
        return loader

    return (
        create_data_loader([X1, X1_lens]),
        create_data_loader([X2, X2_lens]),
        truncate_index,
    )


def create_test(n, test_df_1, test_df_2):
    """
    Generate test set as int 64 matrix
    @param n:
    @param test_df_1:
    @param test_df_2:
    @return:
    """
    # Generate small test based on ground truth
    test_df_1a = pd.DataFrame()
    test_df_1b = pd.DataFrame()

    for i1, i2 in shuffle(list(itertools.combinations(test_df_1.index, 2)))[:n]:
        try:
            test_df_1a = test_df_1a.append(test_df_1.iloc[i1, :])
            test_df_1b = test_df_1b.append(test_df_2.iloc[i2, :])
        except:
            print(i1, i2)

    test_df_1b = test_df_1b.append(test_df_1)
    test_df_1a = test_df_1a.append(test_df_2)

    test_df_1a.reset_index(inplace=True)
    test_df_1b.reset_index(inplace=True)

    return test_df_1a, test_df_1b


def to_cuda(loader, device):
    """
    Transfer your dataloader into CPU or GPU
    @param loader: DataLoader
    @param device: torch.device
    @return: dataloader in specific device
    """
    return [load.to(device) for load in loader]


def validate(model, X1, X2, device):
    y_true = []
    y_pred = []
    dist_list = []
    X1_embed, X2_embed = [], []
    for a, b in zip(X1, X2):
        # Send data to graphic card - Cuda0
        a, b = to_cuda(a, device), to_cuda(b, device)
        with torch.no_grad():
            a, b = model(a, b)
            a, b = a.cpu(), b.cpu()
            a = a.reshape(a.shape[0], -1)
            b = b.reshape(b.shape[0], -1)
            #         att1 = att1.cpu()
            #         att2 = att2.cpu()
            X1_embed.append(a)
            X2_embed.append(b)
            dist = np.array(
                [
                    cosine_similarity([a[i].numpy()], [b[i].numpy()])
                    for i in range(0, len(a))
                ]
            ).flatten()
            dist_list.append(dist)

            y_true_curr = np.zeros(len(dist))
            y_true = np.concatenate([y_true, y_true_curr])

            y_pred_curr = np.ones(len(dist))
            y_pred_curr[np.where(dist < 0.74)[0]] = 0
            y_pred = np.concatenate([y_pred, y_pred_curr])
    y_true[1176:] = 1
    return y_true, y_pred, dist_list, X1_embed, X2_embed


# ---- MAIN
batch_size = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '~/Desktop/GT_added.csv'
model_path = 'pretrain/online_model.h5'

test_df = pd.read_csv(data_path, encoding="ISO-8859-1")
test_df.fillna("", inplace=True)
test_df_1 = test_df.loc[:, ["address"]]
test_df_1["content"] = (
    test_df_1["address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
)
test_df_2 = test_df.loc[:, ["duplicated_address"]]
test_df_2["content"] = (
    test_df_2["duplicated_address"]
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"[ ]+", " ", regex=True)
        .str.replace("null", "")
        .str.replace("nan", "")
)

if __name__ == "__main__":
    test_df_1a, test_df_1b = create_test(1176, test_df_1, test_df_2)
    test_X1, test_X2, test_drop = data_loader(test_df_1a, test_df_1b, default_vocab, batch_size)

    # ---- Load model, distance and optimizer
    # Load model & optimizer
    margin = 0.4
    embeddings_dim = 50
    n_classes = 30
    hid_dim = 50
    layers = 1
    batch_size = 13
    lr = 0.01
    early_stopping_steps = 5
    max_length = 121

    model = biGru(embedding_net=CharacterEmbedding(embeddings_dim, vocab=default_vocab, max_length=max_length),
                  n_classes=n_classes, hid_dim=hid_dim, layers=1).to(device)
    distance = TripletDistance(margin=margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.eval()

    # Test
    y_true, y_pred, _, _, _ = validate(model, test_X1, test_X2, device)
    print(
        "\tAccuracy:\t{}\tF1-score:\t{}\t".format(
            round(accuracy_score(y_true, y_pred), 4),
            round(f1_score(y_true, y_pred), 4),
        ),
        end="",
    )
    print(
        "Precision:\t{}\t\tRecall:\t{}".format(
            round(precision_score(y_true, y_pred), 4),
            round(recall_score(y_true, y_pred), 4),
        ),
        end="",
    )
