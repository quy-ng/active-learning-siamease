import click
import torch
import sys
import codecs
import torch.optim as optim
import os
import json
# Set up the network and training parameters
from models import CharacterEmbedding, biGru
from models.losses import OnlineTripletLoss, TripletDistance
from models.sampling import HardestNegativeTripletSelector
from dataset import Inspectorio
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

model_path = 'pretrain/trained_batch_model_v02.h5'
model_save = 'pretrain/online_model.h5'

n_epochs = 20
log_interval = 50


@click.command()
@click.option('--data_path')
@click.option('--task_id')
def run_online(data_path, task_id):
    print(data_path, task_id)
    task_submit = f'file_{task_id}_submit.json'

    train_dataset, raw_presentation = Inspectorio.load_data(data_path, batch_size, default_vocab)
    model = biGru(embedding_net=CharacterEmbedding(embeddings_dim, vocab=default_vocab, max_length=max_length),
                  n_classes=n_classes, hid_dim=hid_dim, layers=1)

    loss_fn = OnlineTripletLoss(margin,
                                HardestNegativeTripletSelector(margin, cpu=True, is_web=True, task_id=task_id),
                                TripletDistance(margin).to(device), max_length, is_web=True, task_id=task_id)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    loss_list = []
    model.train()

    want_stop = False

    for epoch in range(n_epochs):
        avg_loss = 0
        avg_pos_sim = 0
        avg_neg_sim = 0
        for batch, data in enumerate(train_dataset):
            df_idx = data[2].cpu().numpy()
            x = model(data)
            loss, pos_sim, neg_sim, n_triplets = loss_fn(x, model, (df_idx, raw_presentation[df_idx]), default_vocab)
            # Append to batch list
            avg_loss += float(loss)
            avg_pos_sim += pos_sim.mean()
            avg_neg_sim += neg_sim.mean()
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss /= n_triplets
            avg_pos_sim /= n_triplets
            avg_neg_sim /= n_triplets
            loss_list.append(avg_loss)

            print("Do you want to add more input? \n")
            valid_response = False
            while not valid_response:
                valid_responses = {'y', 'n'}
                if os.path.isfile(task_submit):
                    existing_status = codecs.open(task_submit, 'r', 'UTF-8').read()
                    existing_status = json.loads(existing_status)
                    user_input = existing_status['user_input']
                if user_input in valid_responses:
                    valid_response = True
            if user_input == 'y':
                pass
            if user_input == 'n':
                print('Stopping training', file=sys.stderr)
                want_stop = True
                break

        os.remove(task_submit)

        if want_stop:
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
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                model_save,
            )
            print(f'\nSaved params! at {model_save}', file=sys.stderr)
            break


if __name__ == '__main__':
    run_online()
