import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform= transforms.ToTensor())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle= True)
dataiter = iter(trainloader)
images,labels= dataiter.next()
print(images.shape)

print(images[0].shape)
print(labels[0].item())
img = images[1]
print(type(img))

npimg = img.numpy()
print(npimg.shape)
npimg = np.transpose(npimg, (1,2,0))
print(npimg.shape)
plt.figure(figsize = (1,1))
plt.imshow(npimg)
plt.show()

def imshow(img):
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))

imshow(torchvision.utils.make_grid(images))
print(' '.join(classes[labels[j]] for j in range(4)))

import torch.nn as nn

class FirstCNN(nn.Module):
  def __init__(self):
    super(FirstCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3) 
    

  def forward(self, x):
      x= self.conv1(x)
      return x
net = FirstCNN()

out = net(images)
out.shape
for param in net.parameters():
  print(param.shape)
out1 = out[0,0, :, :].detach().numpy()
print(out1.shape)

plt.imshow(out[0,0, :, :].detach().numpy())
plt.show()

class FirstCNN_v2(nn.Module):
  def __init__(self):
    super(FirstCNN_v2, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(3,8,3),
        nn.Conv2d(8,16,3)
    )

  def forward(self,x):
    x= self.model(x)
    return x

net = FirstCNN_v2()
out = net(images)
out.shape

plt.imshow(out[0 ,0 ,:, :].detach().numpy())

class FirstCNN_v3(nn.Module):
  def __init__(self):
    super(FirstCNN_v3, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(3,6,5),
        nn.AvgPool2d(2,stride=2),
        nn.Conv2d(6,16,5),
        nn.AvgPool2d(2, stride=2)
    )

  def forward(self,x):
    x= self.model(x)
    return x

net = FirstCNN_v3()
out = net(images)
out.shape

plt.imshow(out[0 ,0 ,:, :].detach().numpy())

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.cnn_model = nn.Sequential(
        nn.Conv2d(3,6,5),
        nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        nn.Conv2d(6,16,5),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2)
    )
    self.fc_model = nn.Sequential(
        nn.Linear(400,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10)
    )
  
  def forward(self,x):
    print(x.shape)
    x = self.cnn_model(x)
    print(x.shape)
    x = x.view(x.size(0), -1)
    print(x.shape)
    x= self.fc_model(x)
    print(x.shape)
    return x
net = LeNet()
out = net(images)

print(out)

max_values, pred_class = torch.max(out.data, 1)
print(pred_class)

#Training Net
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.cnn_model = nn.Sequential(
        nn.Conv2d(3,6,5),
        nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        nn.Conv2d(6,16,5),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2)
    )
    self.fc_model = nn.Sequential(
        nn.Linear(400,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10)
    )
  
  def forward(self,x):
    x = self.cnn_model(x)
    x = x.view(x.size(0), -1)
    x = self.fc_model(x)
    return x
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform= transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle= True)
testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform= transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle= True)
def evaluation(dataloader):
  total, correct = 0,0
  for data in dataloader:
    inputs, labels = data
    outputs = net(inputs)
    _ , pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum().item()
  return 100 * correct / total
net = LeNet()
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters())

#time
loss_arr = []
loss_epoch_arr = []
max_epochs = 16

for epoch in range(max_epochs):

  for i,data in enumerate(trainloader,0):

    inputs,labels = data
    opt.zero_grad()

    outputs = net(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    opt.step()

    loss_arr.append(loss.item())

  loss_epoch_arr.append(loss.item())

  print('Epoch: %d / %d, Test acc: %0.2f, Train acc: %0.2f' % (epoch, max_epochs, evaluation(testloader), evaluation(trainloader)))

plt.plot(loss_epoch_arr)
plt.show()
