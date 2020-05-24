import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from tsexp.layers import Covariance
from tsexp.timeseries import StockMarketDataset, WindowedDataset


def plot_folder():
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, '..', 'plots')


def new_plot(name):
    return f'{plot_folder()}/{name}'


def cache_folder():
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, '..', 'cache')


class CovarianceAdapter(nn.Module):
    def __init__(self, input_shape, k=2, channels=None, bias=True):
        super(CovarianceAdapter, self).__init__()
        n, m = input_shape
        self.cov = Covariance(input_shape, k=k, channels=channels, bias=bias)
        s = self.cov.diff.out_channel
        self.adapter = nn.Linear(s * s, n * n)

        if s * s == n * n:
            self.adapter.weight = nn.Parameter(torch.eye(n * n, dtype=torch.float), requires_grad=True)

        self.n = n
        self.size = s * s

    def forward(self, x):
        x = self.cov(x)
        x = x.view(-1, self.size)
        return self.adapter(x).view(-1, self.n, self.n)

    def randomize(self):
        # _ = torch.nn.init.normal_(self.cov.diff.kernel, mean=0.0, std=1.0)
        _ = torch.nn.init.normal_(self.cov.mean.weight, mean=0.0, std=1.0)
        _ = torch.nn.init.normal_(self.cov.quadratic.A, mean=0.0, std=1.0)

    def visualize_diff(self, tickers, namespace):
        k = self.cov.diff.kernel.detach().cpu()
        num, _, _ = k.shape

        for i in range(num):
            fig, ax = plt.subplots()
            im = ax.imshow(k[i, :, :].numpy())

            ax.set_yticks(np.arange(len(tickers)))
            ax.set_yticklabels(tickers)

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('v', rotation=-90, va="bottom")
            fig.savefig(new_plot(f'{namespace}_diff_{i}_{tickers[i % len(tickers)]}.png'))
            plt.close(fig)

    def visualize_quadratic(self, tickers, namespace):
        A = self.cov.quadratic.A.detach().cpu()
        fig, ax = plt.subplots()
        im = ax.imshow(A.numpy())
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('v', rotation=-90, va="bottom")
        fig.savefig(new_plot(f'{namespace}_mat_quadratic.png'))
        plt.close(fig)

        d = A.numpy()
        fig, ax = plt.subplots()
        plt.bar(list(reversed(range(len(A)))), [d[i, i] for i in range(len(A))])
        fig.savefig(new_plot(f'{namespace}_diag_quadratic.png'))
        plt.close(fig)

    def visualize_mean(self, namespace):
        w = self.cov.mean.weight.detach().cpu()
        fig, ax = plt.subplots()
        plt.bar(list(reversed(range(len(w)))), w * 100 / w.sum())
        fig.savefig(new_plot(f'{namespace}_mean.png'))
        plt.close(fig)

    def visualize_adapter(self, namespace):
        w = self.adapter.weight.detach().cpu()
        # figsize=(80, 80)
        fig, ax = plt.subplots()
        im = ax.imshow(w.numpy())
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('v', rotation=-90, va="bottom")
        fig.savefig(new_plot(f'{namespace}_adapter.png'))
        plt.close(fig)

    def viz(self, tickers, namespace, other_graphs=True):
        if other_graphs:
            self.visualize_diff(tickers, namespace)
            self.visualize_quadratic(tickers, namespace)
            self.visualize_adapter(namespace)

        self.visualize_mean(namespace)


class MinVariance(nn.Module):
    def __init__(self, num, estimator):
        super(MinVariance, self).__init__()
        self.cov_estimator = estimator
        self.n = num

    def forward(self, input):
        """Returns the target weight in %"""
        batch_size, _, _ = input.shape
        cov = self.cov_estimator(input)

        A = torch.zeros((batch_size, self.n + 1, self.n + 1))
        A[:, self.n, 0:self.n] = 1
        A[:, 0:self.n, self.n] = 1
        A[:, 0:self.n, 0:self.n] = cov

        B = torch.zeros((batch_size, self.n + 1, 1))
        B[:, self.n] = 1

        x, _ = torch.solve(B, A)
        return x


def train_run(epoch, lr, windowed, model, future, tickers, namespace):
    # _, _, size = x.shape

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    current_loss = float('+inf')
    mse = nn.MSELoss()
    min_loss = float('+inf')
    batch_loss = []
    epoch_loss = []

    for i in range(epoch):
        losses = []
        for s in windowed:
            # Select a time segment to predict
            # s = random.randint(n, 252 - n)

            x_train, x_target = s
            x_train, x_target = x_train.cuda(), x_target.cuda()

            weight_predict = model(x_train)

            with torch.no_grad():
                target = future(x_target)

            loss = mse(weight_predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.detach())
            losses.append(loss.detach())

        prev_loss = current_loss
        current_loss = sum([l.item() for l in losses]) / len(losses)

        epoch_loss.append(current_loss)
        if current_loss < min_loss:
            min_loss = current_loss
            model.cov_estimator.viz(tickers, namespace=f'min_{namespace}', other_graphs=True)

        if i % 100 == 0:
            state = dict(
                model=model.state_dict(),
                loss=current_loss)
            torch.save(state, os.path.join(cache_folder(), f'{namespace}_model.pt'))

        print('\r', i, current_loss, f'diff {current_loss - prev_loss}', end='')
        # draw_loss(epoch_loss, 'epoch')
        # draw_loss(batch_loss, 'batch')

    state = dict(
        model=model.state_dict(),
        loss=current_loss)
    torch.save(state, os.path.join(cache_folder(), f'{namespace}_model.pt'))
    return batch_loss, epoch_loss


def draw_loss(loss, name):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    plt.plot(loss)
    fig.savefig(new_plot(f'{name}_loss.png'))
    plt.close(fig)


def main(n, lags, mult, epochs, lr, seed=0):
    from olympus.utils import set_seeds
    set_seeds(seed)

    # tickers = ['AAPL', 'MSFT', 'GIS', 'TSLA', 'LUV', 'MMM', 'BLK']
    tickers = [
        # 1     2      3     4      5       6     7     8      9    10
        'MO', 'AEP', 'BA', 'BMY', 'CPB', 'CAT', 'CVX', 'KO', 'CL', 'COP',    # 1
        'ED', 'CVS', 'DHI', 'DHR', 'DRI', 'DE', 'D', 'DTE', 'ETN', 'EBAY',   # 2
        'F',  'BEN', 'HSY', 'HBAN', 'IBM', 'K', 'GIS', 'MSI', 'NSC', 'TXN'
    ]
    name = f'n={n}_lags={lags}_mult={mult}_epochs={epochs}_lr={lr}_seed={seed}'

    # 1 quarter is ~60 work days
    cov_estimator = CovarianceAdapter((len(tickers), n), k=lags, channels=len(tickers) * mult).cuda()
    cov_estimator.viz(tickers, namespace=f'init_{name}')

    model = MinVariance(len(tickers), cov_estimator)

    oracle_estimator = Covariance((len(tickers), n), k=2).cuda()
    future = MinVariance(len(tickers), oracle_estimator)

    try:
        state = torch.load(os.path.join(cache_folder(), f'{name}_model.pt'))
        model.load_state_dict(state['model'])
    except FileNotFoundError:
        pass

    cov_estimator.cov.smooth()
    cov_estimator.viz(tickers, namespace=f'start_{name}')

    dataset = StockMarketDataset(tickers, '2000-01-01', '2019-05-10')

    windowed = WindowedDataset(
        dataset,
        window=n,
        transforms=lambda x: x.transpose(1, 0)
    )

    loader = DataLoader(windowed, batch_size=32)

    batch_loss, epoch_loss = train_run(epochs, lr, loader, model, future, tickers, name)

    cov_estimator.viz(tickers, namespace=f'end_{name}')


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--n', default=120, type=int)
    # parser.add_argument('--lags', default=4, type=int)
    # parser.add_argument('--multiplier', default=1, type=int)
    # parser.add_argument('--epochs', default=1000, type=int)
    # parser.add_argument('--lr', default=1e-7, type=int)
    # args = parser.parse_args()

    from multiprocessing import Process

    # main(30, 4, 1, 1000, 1e-7)

    # One months + 10 days
    p1 = Process(target=main, args=(30, 4, 1, 1000, 1e-7, 3328080944))
    # p2 = Process(target=main, args=(130, 4, 1, 1000, 1e-7, 1837592235))
    # p3 = Process(target=main, args=(130, 4, 1, 1000, 1e-7, 2023952334))
    # p4 = Process(target=main, args=(130, 4, 1, 1000, 1e-7, 3939345744))
    # One Quarter + 10 days
    p2 = Process(target=main, args=(70, 4, 1, 1000, 1e-7, 3328080944))
    # two Quarters + 10 days
    p3 = Process(target=main, args=(130, 4, 1, 1000, 1e-7, 3328080944))
    # three Quarters + 10 days
    p4 = Process(target=main, args=(190, 4, 1, 1000, 1e-7, 3328080944))

    ps = [p1, p2, p3, p4]   #
    for p in ps:
        p.start()

    for p in ps:
        p.join()
