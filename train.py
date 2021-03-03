import numpy as np
import torch
import torch.optim as optim
import time
import os
import copy
import argparse
from torch import nn
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from models import ConvBnNet, Resnet


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Net", dest="net",
                        help="model arch (defult ConvBnNet)", type=str)
    parser.set_defaults(net='ConvBnNet')
    parser.add_argument("--num_epochs", dest="epochs",
                        help="num_epochs (defult 10)", type=int)
    parser.set_defaults(epochs=10)
    parser.add_argument("--lr", dest="lr", help="lr (defult 0.001)", type=float)
    parser.set_defaults(lr=0.001)
    parser.add_argument("--save_dir", dest="save_dir",
                        help="save model to", type=str)
    parser.set_defaults(save_dir='model')
    args = parser.parse_args()

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.net == 'ConvBnNet':
        model = ConvBnNet([10, 20, 40], 10)
    elif args.net == 'Resnet':
        model = Resnet([10, 20, 40], 10)
    else:
        raise ValueError("Oops! not valid model arch. ")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)

    torch.save(model.state_dict(), os.path.join("model", args.net + '_torch_model.pt'))
