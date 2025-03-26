import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    """
    CNN architecture for MNIST classification
    Input: 1x28x28 (MNIST images are grayscale 28x28 pixels)
    Output: 10 (probability distribution over 10 digit classes)
    """
    def __init__(self):
        super(Net, self).__init__()
        # Layer 1
        # Input: 1x28x28 (MNIST image)
        # Output: 8x28x28 (same padding keeps dimensions)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Layer 2
        # Input: 8x28x28
        # Output: 16x28x28 (same padding keeps dimensions)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Layer 3
        # Input: 16x14x14 (after maxpool from previous layer)
        # Output: 16x14x14 (same padding keeps dimensions)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # Global Average Pooling
        # Input: 16x14x14
        # Output: 16x1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dropout with 10% probability
        self.dropout = nn.Dropout(0.1)

        # Final fully connected layer
        # Input: 16 (flattened 16x1x1)
        # Output: 10 (one for each digit class)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # x shape: batch_size x 1 x 28 x 28
        x = self.bn1(F.relu(self.conv1(x)))  # -> batch_size x 8 x 28 x 28
        x = self.bn2(F.relu(self.conv2(x)))  # -> batch_size x 16 x 28 x 28
        x = F.max_pool2d(x, 2)  # -> batch_size x 16 x 14 x 14
        x = self.dropout(x)
        
        x = self.bn3(F.relu(self.conv3(x)))  # -> batch_size x 16 x 14 x 14
        x = self.gap(x)  # -> batch_size x 16 x 1 x 1
        x = x.view(-1, 16)  # -> batch_size x 16
        x = self.fc(x)  # -> batch_size x 10
        return F.log_softmax(x, dim=1)  # -> batch_size x 10 (probabilities)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    batch_size = 128
    epochs = 15
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Split training data into train and validation (50k/10k)
    full_train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, val_loader)
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "mnist_model.pt")
        
        print(f"Best accuracy so far: {best_accuracy:.2f}%")
    
    # Load best model and test
    model.load_state_dict(torch.load("mnist_model.pt"))
    final_accuracy = test(model, device, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    
    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {param_count}")

if __name__ == '__main__':
    main() 