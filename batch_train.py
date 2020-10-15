import time
from tqdm import tqdm

import torch
import torch.optim as optim

# Set up the network and training parameters
from models import CharacterEmbedding, biGru
from models.losses import TripletDistance
from dataset import InspectorioLabel
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

model_path = './pretrain/trained_batch_model.h5'

anc_loader, pos_loader, neg_loader, max_length = InspectorioLabel.load_data(
        './data/dac/dedupe-project/new/new_generated_labeled_data.csv', batch_size, default_vocab
    )

print(f'dataset max length is {max_length}')

model = biGru(embedding_net=CharacterEmbedding(embeddings_dim, vocab=default_vocab, max_length=max_length),
              n_classes=n_classes, hid_dim=hid_dim, layers=1)
if cuda:
    model.cuda()
loss_fn = TripletDistance(margin).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

n_epochs = 20
log_interval = 50

if __name__ == '__main__':

    # ---- Train model
    best_lost = None
    loss_list = []
    model.train()
    start_time = time.time()
    for epoch in tqdm(range(n_epochs), desc="Epoch", total=n_epochs):
        avg_loss = 0
        avg_pos_sim = 0
        avg_neg_sim = 0
        for batch, [anc_x, pos_x, neg_x] in enumerate(
                zip(anc_loader, pos_loader, neg_loader)
        ):
            # Send data to graphic card - Cuda
            anc_x, pos_x, neg_x = (
                to_cuda(anc_x, device),
                to_cuda(pos_x, device),
                to_cuda(neg_x, device),
            )
            # Load model and measure the distance between anchor, positive and negative
            x, pos, neg = model(anc_x, pos_x, neg_x)
            loss, pos_sim, neg_sim = loss_fn(x, pos, neg)
            # Append to batch list
            avg_loss += float(loss)
            avg_pos_sim += pos_sim.mean()
            avg_neg_sim += neg_sim.mean()
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Average loss and distance of all epochs
        avg_loss /= len(anc_loader)
        avg_pos_sim /= len(anc_loader)
        avg_neg_sim /= len(anc_loader)
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
    print("--- %s seconds ---" % (time.time() - start_time))
