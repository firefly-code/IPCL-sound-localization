import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import csv 
from pathlib import Path
import numpy as np
import re
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class CochleagramLoader(Dataset):
    def __init__(self,coch_dir, transform=None, target_transform=None):
        self.coch_dir = coch_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.coch_dir)

    def __getitem__(self, idx):
        image = np.load(str(self.coch_dir[idx]))['arr_0']
        image_torch = torch.from_numpy(image).float()
        image_torch = image_torch
        pattern = r"Az_(?P<azimuth>-?\d+)_El_(?P<elevation>-?\d+)"
        labelData = re.search(pattern, str(self.coch_dir[idx]))
        azimuth = int(labelData.group('azimuth'))
        elevation = int(labelData.group('elevation'))
        label = 0
        with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                if(int(row[0])==azimuth and int(row[1])==elevation):
                    label = row[2]
                    break
        label = torch.from_numpy(np.array(int(label)))
        label = label.cuda()
        if self.transform:
            image_torch = self.transform(image_torch)
        if self.target_transform:
            label = self.target_transform(label)
        return image_torch, label


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels,  stride: int = 1, expansion: int = 1, identity_downsample= None
    ):
        super(block,self).__init__()
        self.expansion = expansion
        self.identity_downsample= identity_downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.elu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        # self.elu2 = nn.ELU()
        # self.conv3 = nn.Conv2d(
        #     intermediate_channels,
        #     intermediate_channels * self.expansion,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        # )
        # self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        # self.elu3 = nn.ELU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.elu1(out)
   
        out = self.conv2(out)
     
        out = self.bn2(out)
  
        # x = self.elu2(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(x)
        out += identity
        out = self.elu1(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, sound_channel, num_classes):
        super(ResNet, self).__init__()
        self.expansion= 1
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=sound_channel, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.elu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)

        x = self.elu(x)

        x = self.maxpool(x)
       
        x = self.layer1(x)

        x = self.layer2(x)
        
        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride: int = 1):
        identity_downsample = None
        
        self.expansion = 1
        print("stride",stride)
        print("in_chan",self.in_channels)
        if stride != 1 or self.in_channels != intermediate_channels * 1:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, intermediate_channels, stride,self.expansion, identity_downsample)
        )

        self.in_channels = intermediate_channels * self.expansion


        for i in range(num_residual_blocks - 1):
            layers.append(block(in_channels = self.in_channels, intermediate_channels=intermediate_channels,expansion = self.expansion))

        return nn.Sequential(*layers)


def ResNet34(sound_channel, num_classes):
    return ResNet(block, [3, 4, 6, 3], sound_channel, num_classes)

def ResNet18(sound_channel, num_classes):
    return ResNet(block, [2, 2, 2, 2], sound_channel, num_classes)


def train_one_epoch(epoch_index, tb_writer,train_dataloader,optimizer,model,loss_fn):
    running_loss = 0.
    running_acc =0.
    last_loss = 0.
    data_checkpoint = 10
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        _, prediction = torch.max(outputs.data,1)
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        running_acc += (labels == prediction).sum().item()
        if i % data_checkpoint == data_checkpoint-1:
            last_loss = running_loss / data_checkpoint # loss per batch
            last_acc = running_acc / (data_checkpoint*len(prediction))
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Acc/train', last_acc, tb_x)
            running_loss = 0.
            running_acc = 0.

    return last_loss




def RunTraining():
    
    net = ResNet34(sound_channel=2, num_classes=190).to('cuda')
    
    learning_rate = 0.01
    dataset_train = CochleagramLoader(list(Path("Data/Train").glob('*.npz')))
    dataset_val = CochleagramLoader(list(Path("Data/Val").glob('*.npz')))

    
    ##define the test and training sets
    
    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=True)
    # for sounds, labels in train_dataloader:
    #     print("Batch of images has shape: ",sounds.shape)
    #     print("Batch of labels has shape: ", labels.shape)
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
    
    

    # Train the model
    total_step = len(train_dataloader)
    tb_writer = SummaryWriter()
    
    dataiter = iter(train_dataloader)
    sounds, labels = next(dataiter)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/resnet_ssl{}'.format(timestamp))
    
    writer.add_graph(net,sounds)
    
    epoch_number = 0

    EPOCHS = 2534

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)  
        avg_loss = train_one_epoch(epoch_number, tb_writer,train_dataloader,optimizer,net,loss_fn)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        net.eval()
        running_vacc = 0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_dataloader):
                vinputs, vlabels = vdata
                voutputs = net(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                _, predictionV = torch.max(voutputs.data,1)
                running_vacc += (vlabels == predictionV).sum().item()
                running_vloss += vloss
        last_acc = running_vacc / ((i + 1)*len(predictionV))
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        tb_writer.add_scalar('Acc/Val', last_acc, epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(net.state_dict(), model_path)

        epoch_number += 1
def RunTestSet():
    model = ResNet34(sound_channel=2, num_classes=190).to('cpu')
    model.load_state_dict(torch.load("model_20240509_003608_24"))
    model.eval()
    y_vs_yhat = []
    dataset_val = CochleagramLoader(list(Path("Data/Val").glob('*.npz')))
    validation_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=True)
    for image,label in list(validation_dataloader):
        with torch.no_grad():
          pred = model(image)
          _, predictionV = torch.max(pred.data,1)
          y_vs_yhat.append([label.cpu().numpy(),predictionV.cpu().numpy()])
    np.save("val_data2",np.array(y_vs_yhat))
        

if __name__ == "__main__":
    #RunTraining()
    RunTestSet()