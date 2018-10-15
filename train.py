from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(32*9*2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 32*9*2)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4

if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ]),
    }

    data_dir = 'data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0")


    model = Net()
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=40)
    torch.save(model, 'torch_model.pkl')
