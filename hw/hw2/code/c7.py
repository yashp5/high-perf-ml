import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import time

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.linear = nn.Linear(512, 10)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to CIFAR-10 dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use (sgd, sgd_nesterov, adagrad, adadelta, adam)')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # DataLoader for CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    # Model, loss, and optimizer
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer.lower() == 'sgd_nesterov':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=5e-4)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=5e-4)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    else:
        raise ValueError('Unsupported optimizer. Use "sgd", "sgd_nesterov", "adagrad", "adadelta", or "adam".')

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Start timing for data loading
        data_loading_start_time = time.perf_counter()
        data_loading_time = 0.0
        training_time = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            data_loading_end_time = time.perf_counter()
            data_loading_time += data_loading_end_time - data_loading_start_time

            # Start timing for training
            training_start_time = time.perf_counter()

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_end_time = time.perf_counter()
            training_time += training_end_time - training_start_time

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/(batch_idx+1):.4f}, Accuracy: {100.*correct/total:.2f}%')

            # Reset data loading timer
            data_loading_start_time = time.perf_counter()

        # Total epoch time
        epoch_end_time = time.perf_counter()
        total_epoch_time = data_loading_time + training_time

        print(f'Epoch [{epoch+1}/{num_epochs}] Summary: Data Loading Time: {data_loading_time:.4f}s, Training Time: {training_time:.4f}s, Total Epoch Time: {total_epoch_time:.4f}s')

if __name__ == '__main__':
    main()
