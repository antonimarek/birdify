import numpy as np
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self, signal: dict = None, fft: dict = None, fbank: dict = None, mfccs: dict = None):
        self.signal = signal
        self.fft = fft
        self.fbank = fbank
        self.mfccs = mfccs

    def plot_prepare(self, plot_size: tuple, key_list: list, x_arg: list, y_arg: list = False,
                     title: str = 'Time series', n_col: int = 2):
        item_num = len(key_list)
        n_row = int(np.ceil(item_num / n_col))

        fig, _ = plt.subplots(figsize=plot_size, sharey='all')
        fig.suptitle(title, size=16)
        i = 0
        for x in range(n_row):
            for y in range(n_col):
                if i < item_num:
                    ax = plt.subplot2grid((n_row, n_col), (x, y))
                    if y_arg:
                        ax.plot(x_arg[i], y_arg[i])
                    else:
                        ax.plot(x_arg[i])
                    ax.set_title(key_list[i])
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    i += 1
        return self

    def img_prepare(self, plot_size: tuple, key_list: list, images: list,
                    title: str = None, n_col: int = 2):
        item_num = len(key_list)
        n_row = int(np.ceil(item_num / n_col))

        fig, _ = plt.subplots(figsize=plot_size, sharey='all')
        fig.suptitle(title, size=16)
        i = 0
        for x in range(n_row):
            for y in range(n_col):
                if i < item_num:
                    ax = plt.subplot2grid((n_row, n_col), (x, y))
                    ax.imshow(images[i], cmap='hot', interpolation='nearest')
                    ax.set_title(key_list[i])
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    i += 1
        return self

    def signal_plot(self, plot_size: tuple = (20, 5), n_col: int = 2):
        key_list = list(self.signal.keys())
        x_arg = [self.signal[key] for key in key_list]
        self.plot_prepare(plot_size, key_list, x_arg, n_col=n_col)
        plt.show()
        pass

    def fft_plot(self, plot_size: tuple = (20, 5), n_col: int = 2):
        key_list = list(self.fft.keys())
        x_arg = [self.fft[key][1] for key in key_list]
        y_arg = [self.fft[key][0] for key in key_list]
        self.plot_prepare(plot_size, key_list, x_arg, y_arg, n_col=n_col, title='Fourier Transforms')
        plt.show()
        pass

    def fbank_show(self, plot_size: tuple = (20, 5), n_col: int = 2):
        key_list = list(self.fbank.keys())
        images = [self.fbank[key] for key in key_list]
        self.plot_prepare(plot_size, key_list, images, n_col=n_col, title='Filter Bank Coefficients')
        plt.show()
        pass

    def mfccs_show(self, plot_size: tuple = (20, 5), n_col: int = 2):
        key_list = list(self.mfccs.keys())
        images = [self.mfccs[key] for key in key_list]
        self.plot_prepare(plot_size, key_list, images, n_col=n_col, title='Filter Bank Coefficients')
        plt.show()
        pass