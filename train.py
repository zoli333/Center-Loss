# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import ConvNet
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.linalg import norm
import os

# https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/


class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat=2):
        super(CenterLoss, self).__init__()
        self.feat = feat
        self.centers = nn.Parameter(torch.randn(size=(num_classes, feat)).cuda())
        # input [batch_size, feat]
        # labels [batch_size, num_classes]
        # centers [num_classes, feat]

    def forward(self, x, labels):
        batch_size = x.size(0)
        labels = labels.unsqueeze(-1).expand(batch_size, self.feat)
        coords = torch.gather(self.centers, dim=0, index=labels)
        loss = (x - coords).pow(2).sum() / batch_size / 2.0
        return loss


# Define relevant variables for the ML task
batch_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 150
lam = 1.0

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = True)


test_dataset = torchvision.datasets.MNIST(root = './data',
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                          download=True)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)


model = ConvNet(num_classes).to(device)

#Setting the loss function
nllloss = nn.NLLLoss()
centerloss = CenterLoss()

# Train with Adam optimizer
# optimizer4nn = optim.Adam(model.parameters(), lr=0.001)
# optimizer4center = optim.Adam(centerloss.parameters(), lr=0.001)

optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = lr_scheduler.StepLR(optimizer4nn, 100, gamma=0.1)

optimizer4center = optim.SGD(centerloss.parameters(), lr=0.5)

total_step = len(train_loader)

def visualize_neurons(all_features, step, viz_folder):
    os.makedirs(viz_folder, exist_ok=True)
    with torch.no_grad():
        x = torch.cat(all_features, 0)
        x = x.cpu().numpy()
        x_ = x[:, 0]
        y_ = x[:, 1]
        fig = plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
        plt.scatter(x_, y_)
        plt.axis('off')
        plt.savefig(viz_folder + "/result_" + str(step) + ".png")
        plt.close(fig)


def test(model, step):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        all_features = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, features = model(images)
            all_features.append(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        visualize_neurons(all_features, step, "viz_test")


if __name__ == "__main__":
    step = 0
    for epoch in range(num_epochs):
        all_features = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, fc1 = model(images)
            all_features.append(fc1)
            cross_entropy_loss = nllloss(outputs, labels)
            center_loss = centerloss(fc1, labels)
            loss = cross_entropy_loss + lam * center_loss

            # Backward and optimize
            optimizer4nn.zero_grad()
            optimizer4center.zero_grad()

            loss.backward()

            optimizer4nn.step()
            optimizer4center.step()

            if (i + 1) % total_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if (epoch + 1) % 10 == 0:
            step = epoch + 1
            test(model, step)
            visualize_neurons(all_features, step, "viz_train")

        scheduler.step()
