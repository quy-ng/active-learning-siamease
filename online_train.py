import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm

# Set up the network and training parameters
from models import CharacterEmbedding, biGru
from models.losses import OnlineTripletLoss, TripletDistance
from models.sampling import HardestNegativeTripletSelector
from models.metrics import AverageNonzeroTripletsMetric
from dataset import Inspectorio
from trainer import fit
from ultils import to_cuda
from ultils.character_level import default_vocab

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

margin = 0.4
embeddings_dim = 50
n_classes = 30
hid_dim = 50
layers = 1
batch_size = 13
lr = 0.01
early_stopping_steps = 5
max_length = 121

train_dataset, raw_presentation = Inspectorio.load_data('~/Desktop/active_learning_data.xlsx', batch_size, default_vocab)
model_path = 'pretrain/trained_batch_model_v02.h5'

model = biGru(embedding_net=CharacterEmbedding(embeddings_dim, default_vocab, max_length),
              n_classes=n_classes, hid_dim=hid_dim, layers=1)
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin,
                            HardestNegativeTripletSelector(margin, cpu=True),
                            TripletDistance(margin).to(device))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

if __name__ == '__main__':
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    best_lost = None
    loss_list = []
    model.train()
    for epoch in tqdm(range(n_epochs), desc="Epoch", total=n_epochs):
        avg_loss = 0
        avg_pos_sim = 0
        avg_neg_sim = 0
        for batch, data in enumerate(train_dataset):
            df_idx = data[2].cpu().numpy()
            x = model(data)
            loss = loss_fn(x, model, (df_idx, raw_presentation[df_idx]), default_vocab)
            # Append to batch list
            avg_loss += float(loss)
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Average loss and distance of all epochs
        avg_loss /= len(train_dataset)
        loss_list.append(avg_loss)
        print(
            "\rEpoch:\t{}\tAverage Loss:\t{}\t\tPos:\t{}\t\tNeg:\t{}\t\t".format(
                epoch,
                round(avg_loss, 4),
                round(float(avg_pos_sim), 4),
                round(float(avg_neg_sim), 4),
            ),
            end="",
        )
        # Save model thought each checkpoint
        # Early stopping after reachs {early_stopping_steps} steps
        forward_index = 0
        if best_lost is None or best_lost > avg_loss:
            best_lost = avg_loss
            forward_index = 0
            # Save checkpoint every time we get the better loss
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                model_path,
            )
        else:
            forward_index += 1
            if forward_index == early_stopping_steps or best_lost == 0:
                break
