import torch
from torch.utils.data import DataLoader

# Set up the network and training parameters
from models import CharacterEmbedding, biGru
from models.losses import OnlineTripletLoss
from models.sampling import HardestNegativeTripletSelector
from models.metrics import AverageNonzeroTripletsMetric

from trainer import fit

cuda = torch.cuda.is_available()
if cuda:
    cpu = False
else:
    cpu = True
from torch.optim import lr_scheduler
import torch.optim as optim

from dataset import Inspectorio
from dataset.augmentation import augment_dataframe

margin = 1.
embeddings_dim = 50
n_classes = 30
hid_dim = 50
layers = 1
batch_size = 13

train_dataset = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=None)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = DataLoader(train_dataset, **kwargs, batch_size=batch_size)

model = biGru(embedding_net=CharacterEmbedding(embeddings_dim),
              n_classes=n_classes, hid_dim=hid_dim, layers=1)
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin, cpu=cpu))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

if __name__ == '__main__':
    fit(online_train_loader, None, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        metrics=[AverageNonzeroTripletsMetric()])
