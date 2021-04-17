import numpy as np
import matplotlib.pyplot as plt
import seaborn


def class_distribution():
    pass


# GENERALIZE CONFUSION MATRIX
# code by Mehmet Emin Yıldırım
# from https://medium.com/analytics-vidhya/how-to-generalize-a-multi-class-confusion-matrix-912e29284553
def generalize_cfmatrix(cf_matrix, num_groups):
    gen_cfmatrix = np.zeros((num_groups, num_groups))
    T = np.zeros(num_groups)

    for col in range(len(cf_matrix)):
        for row in range(len(cf_matrix[0])):
            # find gen_i, gen_j
            for k in range(1, num_groups + 1):
                if len(cf_matrix) / num_groups * k > col >= len(cf_matrix) / num_groups * (k - 1):
                    gen_i = k - 1
                if len(cf_matrix) / num_groups * k > row >= len(cf_matrix) / num_groups * (k - 1):
                    gen_j = k - 1
            gen_cfmatrix[gen_i][gen_j] += cf_matrix[col][row]
            if col == row:
                T[gen_i] += cf_matrix[col][row]

    # distribute false values to neighbor columns
    for col in range(len(gen_cfmatrix)):
        for row in range(len(gen_cfmatrix[0])):
            if col == row:
                F = gen_cfmatrix[col][row] - T[col]
                gen_cfmatrix[col][row] -= F
                if row == 0:
                    gen_cfmatrix[col][row + 1] += F
                elif row == len(gen_cfmatrix) - 1:
                    gen_cfmatrix[col][row - 1] += F
                else:
                    F1 = int(F / 2)
                    F2 = F - F1
                    gen_cfmatrix[col][row - 1] += F1
                    gen_cfmatrix[col][row + 1] += F2
    return gen_cfmatrix


def plot_cfm(cf_matrix, labels, title="Confusion Matrix", size=(15, 12)):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=size)
    plt.title(title, {'fontsize': 15, 'fontweight': 'bold'})
    seaborn.set(font_scale=1.1)
    ax = seaborn.heatmap(cf_matrix, annot=True, cmap="YlGnBu", fmt='g')
    l_dict = {'fontsize': 10, 'fontweight': '100'}
    ax.set_xticklabels(labels, l_dict, rotation=90)
    ax.set_yticklabels(labels, l_dict, rotation=0)

    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.show()
    pass
