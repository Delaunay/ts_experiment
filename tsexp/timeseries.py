import hashlib
import os

import torch
from torch.utils.data import Dataset


class StockMarketDataset(Dataset):
    @staticmethod
    def cache_folder():
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, '..', 'cache')

    @staticmethod
    def cache_key(*args):
        m = hashlib.sha256()

        for arg in args:
            if isinstance(arg, list):
                arg = StockMarketDataset.cache_key(*arg)
            m.update(arg.encode('utf-8'))

        return m.hexdigest()[:16]

    @staticmethod
    def fetch_data(tickers, start, end, source):
        import pandas as pd
        from pandas_datareader import data

        folder = StockMarketDataset.cache_folder()

        key = StockMarketDataset.cache_key(tickers, start, end, source)
        cache_file = os.path.join(folder, key)

        if os.path.exists(cache_file):
            aapl = pd.read_csv(cache_file, index_col='Date')
        else:
            aapl = data.DataReader(
                tickers,
                start=start,
                end=end,
                data_source=source)['Adj Close']

            aapl.to_csv(cache_file)

        cleaned = aapl.dropna()
        x = torch.from_numpy(cleaned.values)
        x = x.float().log()

        return x, cleaned

    def __init__(self, tickers, start_date, end_date, source='yahoo'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.data, self.raw = self.fetch_data(tickers, start_date, end_date, source)

    def __len__(self):
        """Returns the number of days inside our time series"""
        return self.data.shape[0]

    def __getitem__(self, item):
        """Returns an array of observation for a given day"""
        if isinstance(item, slice) or (isinstance(item, tuple) and isinstance(item[0], slice)):
            return self.data[item]

        return self.data[item, :]


def identity(x):
    return x


class WindowedDataset(Dataset):
    """Moving window dataset with overlapping observations"""
    def __init__(self, dataset, window=7, transforms=None):
        self.dataset = dataset
        self.window = window

        if transforms is None:
            transforms = identity

        self.transforms = transforms

    def __len__(self):
        return len(self.dataset) - self.window * 2

    def __getitem__(self, item):
        """Returns two tensor representing n days before item and n days after item"""
        item += self.window

        s = item - self.window
        e = item + self.window

        x = self.dataset[s:item, :]
        y = self.dataset[item:e, :]

        return self.transforms(x), self.transforms(y)


if __name__ == '__main__':
    tickers = [
        # 1     2      3     4      5       6     7     8      9    10
        'MO', 'AEP', 'BA', 'BMY', 'CPB', 'CAT', 'CVX', 'KO', 'CL', 'COP',   # 1
        'ED', 'CVS', 'DHI', 'DHR', 'DRI', 'DE', 'D', 'DTE', 'ETN', 'EBAY',  # 2
        'F', 'BEN', 'HSY', 'HBAN', 'IBM', 'K', 'GIS', 'MSI', 'NSC', 'TXN'   # 3
    ]

    dataset = StockMarketDataset(tickers, '2000-01-01', '2019-05-10')

    print(dataset.data.shape)
    print(len(dataset.data))
    print(dataset[0].shape)

    windowed = WindowedDataset(dataset, lags=7, transforms=lambda x: x.transpose(1, 0))

    print(windowed[8][0].shape)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        windowed,
        batch_size=32)

    for batch in loader:
        print(batch[0].shape)
        break



