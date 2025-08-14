
"""
Test GPU's parallel training using a pre-trained ResNet-18 on CIFAR-10 dataset.
"""

import os 
import time
import torch 
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn

print("The number of cuda device: ", torch.cuda.device_count())

# devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
devices = [torch.device("mps")]

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="GPU_Test/", train=is_train,
                                           download=True, transform=augs)
    dataloader = data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=0)
    return dataloader


def evaluate_accuracy_gpu(net, data_iter, device=None): 
    """ Evaluate the accuracy of a model on the GPU."""
    if isinstance(net, nn.Module):
        net.eval() 
        if not device:
            device = next(iter(net.parameters())).device
    metric = [0, 0]
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric[0] += (net(X).argmax(dim=1) == y).sum().float()
            metric[1] += y.numel()
    return metric[0] / metric[1]


def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs."""
    X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = (pred.argmax(dim=1) == y).sum().float()
    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices):
    """Train a model with multiple GPUs"""
    print("Training begin.")
    print("Devices: ", str(devices))

    total_time = 0
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        t = time.time()
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = [0, 0, 0, 0]
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric[0] += l.item()
            metric[1] += acc.item()
            metric[2] += labels.shape[0]
            metric[3] += labels.numel()
        test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0])
        time_epoch = time.time() - t
        total_time += time_epoch
        print(f'epoch {epoch + 1}:')
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f} '
              f'time {time_epoch:.1f} sec')
    
    test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0])
    print("Training complete.")
    print(f'Total time: {total_time:.1f} sec')
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')


def train_fine_tuning(net, learning_rate, batch_size=256, num_epochs=10):
    train_iter = load_cifar10(is_train=True, augs = train_augs, batch_size=batch_size)
    test_iter = load_cifar10(is_train=False, augs = test_augs,  batch_size=batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")

    params_1x = [param for name, param in net.named_parameters()
        if name not in ["fc.weight", "fc.bias"]]
    trainer = torch.optim.SGD([{'params': params_1x},
                                {'params': net.fc.parameters(),
                                'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    
    train(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

train_augs = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
test_augs = transforms.ToTensor()


finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 10)
nn.init.xavier_uniform_(finetune_net.fc.weight)

train_fine_tuning(finetune_net, 5e-5)

