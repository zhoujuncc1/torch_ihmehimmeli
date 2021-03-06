from __future__ import print_function
import argparse
import torch
from torch.cuda.memory import memory_stats_as_nested_dict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from Layer import Linear, cross_entropy_loss, LayerParam

class Net(nn.Module):
    def __init__(self, n_pulse, layer_param):
        super(Net, self).__init__()
        self.layer_param = layer_param
        self.fc1 = Linear(784, 340, n_pulse, layer_param)
        self.fc2 = Linear(340, 10, n_pulse, layer_param)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch, penalty_output_spike_time=0):
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, onehot_target = 1-torch.flatten(data, start_dim=1),  F.one_hot(target, 10)
        data, onehot_target, target = data.to(device), onehot_target.to(device), target.to(device)
        data = torch.where(data==1.0, model.layer_param.kNoSpike, data)
        output = model(data)
        correct += (output.argmin(dim=-1)==target).sum()
        loss = cross_entropy_loss(output, onehot_target, penalty_output_spike_time).mean()
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Acc: {:.4f}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss/(batch_idx+1), correct.item()/(batch_idx+1)/len(target)), end='\r')
            if args.dry_run:
                break


def test(model, device, test_loader, penalty_output_spike_time=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, onehot_target = 1-torch.flatten(data, start_dim=1),  F.one_hot(target, 10)
            data, onehot_target, target = data.to(device), onehot_target.to(device), target.to(device)
            data = torch.where(data==1.0, model.layer_param.kNoSpike, data)
            output = model(data)
            test_loss += cross_entropy_loss(output, onehot_target, penalty_output_spike_time).sum().item()  # sum up batch loss
            pred = output.argmin(dim=-1)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 5)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') 
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--restore', type=str, default=None,
                        help='checkpoint to restore from')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    dataset1 = datasets.MNIST('/home/jun/data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/home/jun/data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    n_pulse = 10
    kNoSpike=1000
    threshold = 1.16732

    decay_rate = 0.18176949150701854
    penalty_no_spike = 48.3748
    pulse_init_multiplier = 7.83912
    nopulse_init_multiplier = -0.275419
    input_range = (0,1)
    layer_param = LayerParam(kNoSpike, decay_rate, threshold, penalty_no_spike, pulse_init_multiplier, nopulse_init_multiplier, input_range, dtype=torch.float, device=device)
    model = Net(n_pulse, layer_param)
    if args.restore:
        model.load_state_dict(torch.load(args.restore))
    model=model.to(device)
    optimizer = optim.Adam([
                {'params': model.fc1.weight},
                {'params': model.fc2.weight},
                {'params': model.fc1.pulse, 'lr': 5.95375e-2},
                {'params': model.fc2.pulse, 'lr': 5.95375e-2}
            ], lr=2.01864e-4)
    # optimizer = optim.Adam(model.parameters(), lr=5.95375e-2)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    if args.restore:
        print("Restored Accuracy:", end=" ")
        best = test(model, device, test_loader)
    else:
        best = 0

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        if args.save_model and acc>=best:
            torch.save(model.state_dict(), "mnist_mlp_best.pt")
            best = acc
#        scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_mlp_last.pt")



if __name__ == '__main__':
    main()
