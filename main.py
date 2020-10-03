import torch

# Set up the network and training parameters
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from models.sampling import AllTripletSelector, \
    HardestNegativeTripletSelector, \
    RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric


from trainer import fit
cuda = torch.cuda.is_available()
from torch.optim import lr_scheduler
import torch.optim as optim

from datasets import BalancedBatchSampler

from dataset import Inspectorio
from dataset.augmentation import augment

train_dataset = Inspectorio('~/Desktop/active_learning_data.xlsx', transform=augment)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

if __name__ == '__main__':
    fit(online_train_loader, None, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        metrics=[AverageNonzeroTripletsMetric()])