import numpy as np
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self, signal: dict, fft: dict):
        self.signal = signal
        self.fft = fft

    def plot_prepare(self, key_list: list, data: dict, title: str = 'Time series'):
        item_num = len(key_list)
        n_col = 2
        n_row = int(np.ceil(item_num / n_col))

        fig, _ = plt.subplots(figsize=(20, 5))
        fig.suptitle(title, size=16)
        i = 0
        for x in range(n_row):
            for y in range(n_col):
                if i < item_num:
                    ax = plt.subplot2grid((n_row, n_col), (x, y))
                    ax.plot(data[key_list[i]])
                    ax.set_title(key_list[i])
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    i += 1
        return self

    def signal_plot(self):
        key_list = list(self.signal.keys())
        self.plot_prepare(key_list, self.signal)
        plt.show()
        pass

    def fft_plt(self):
        return self
