import os

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from tsexp.layers import Covariance
import hashlib


def plot_folder():
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, '..', 'plots')


def new_plot(name):
    return f'{plot_folder()}/{name}'


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

    def visualize_diff(self, tickers):
        k = self.cov.diff.kernel.detach().cpu()

        num, _, _ = k.shape

        for i in range(num):
            fig, ax = plt.subplots()
            im = ax.imshow(k[i, :, :].numpy() / k[i, :, :].sum())

            ax.set_yticks(np.arange(len(tickers)))
            ax.set_yticklabels(tickers)

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('v', rotation=-90, va="bottom")
            fig.savefig(new_plot(f'diff_{i}_{tickers[i % len(tickers)]}.png'))

    def visualize_quadratic(self, tickers):
        A = self.cov.quadratic.A.detach().cpu()
        print(A.shape)
        # print(A)
        fig, ax = plt.subplots()
        im = ax.imshow(A.numpy())
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('v', rotation=-90, va="bottom")

        d = A.numpy()
        fig = plt.bar(range(len(A)), [d[i, i] for i in range(len(A))])
        fig.savefig(new_plot('quadratic.png'))

    def visualize_mean(self):
        w = self.cov.mean.weight.detach().cpu()
        print(w.shape)
        print(w)

        fig = plt.bar(range(len(w)), w * 100 / w.sum())
        fig.savefig(new_plot('mean.png'))

    def visualize_adapter(self):
        w = self.adapter.weight.detach().cpu()
        # figsize=(80, 80)
        fig, ax = plt.subplots()
        im = ax.imshow(w.numpy())
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('v', rotation=-90, va="bottom")
        fig.savefig(new_plot('adapter.png'))

    def viz(self, tickers):
        self.visualize_mean()
        self.visualize_diff(tickers)
        self.visualize_quadratic(tickers)
        self.visualize_adapter()


def cache_key(tickers, start, end):
    m = hashlib.sha256()

    for t in tickers:
        m.update(t.encode('utf-8'))

    m.update(start.encode('utf-8'))
    m.update(end.encode('utf-8'))
    return m.hexdigest()[:16]


def cache_folder():
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, '..', 'cache')


def fetch_data(tickers, start, end):
    import pandas as pd
    from pandas_datareader import data

    folder = cache_folder()

    key = cache_key(tickers, start, end)
    cache_file = os.path.join(folder, key)

    if os.path.exists(cache_file):
        aapl = pd.read_csv(cache_file)
    else:
        aapl = data.DataReader(
            tickers,
            start=start,
            end=end,
            data_source='yahoo')['Adj Close']

        aapl.to_csv(cache_file)

    cleaned = aapl.dropna()
    x = torch.from_numpy(cleaned.values.transpose()).unsqueeze(0)
    x = x.float().cuda().log()

    return x, cleaned


def train_run(epoch, lr, x, n, model, future):
    _, _, size = x.shape

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    current_loss = float('+inf')
    mse = nn.MSELoss()

    for i in range(epoch):
        losses = []
        for s in range(n, size - n):
            # Select a time segment to predict
            # s = random.randint(n, 252 - n)

            x_train = x[:, :, s-n:s]
            x_target = x[:, :, s:s+n]

            cov = model(x_train)

            with torch.no_grad():
                target_cov = future(x_target)

            loss = mse(cov, target_cov)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())

        prev_loss = current_loss
        current_loss = sum([l.item() for l in losses]) / len(losses)

        print('\r', i, current_loss, f'diff {current_loss - prev_loss}', end='')


def main():
    tickers = ['AAPL', 'MSFT', 'GIS', 'TSLA', 'LUV', 'MMM', 'BLK']

    n = 30
    model = CovarianceAdapter((7, n), k=4, channels=7).cuda()
    future = Covariance((7, n), k=2).cuda()

    x, df = fetch_data(tickers, '2000-01-01', '2019-05-10')

    train_run(50, 1e-2, x, n, model, future)

    model.viz(tickers)


if __name__ == '__main__':
    main()
