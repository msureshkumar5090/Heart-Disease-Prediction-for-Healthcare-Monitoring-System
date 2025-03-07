import numpy as np
import warnings
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
warnings.filterwarnings("ignore")

no_of_dataset = 4


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RSA', 'EVO', 'GSOA', 'GOA', 'MGOA']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report Dataset ', i + 1,
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, marker='o', markerfacecolor='red',
                 markersize=5,
                 label='RSA')
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, marker='o', markerfacecolor='green',
                 markersize=5, label='EVO')  # c
        plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, marker='o', markerfacecolor='cyan',
                 markersize=5, label='GSOA')
        plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, marker='o', markerfacecolor='magenta',
                 markersize=5, label='GOA')  # y
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='o', markerfacecolor='black',
                 markersize=5, label='MGOA')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (i + 1))
        plt.show()


def ROC_curve():
    lw = 2
    cls = ['Autoencoder', 'TCNN', 'LSTM', 'CapsNet', 'HDRLSTM-DCN']
    for a in range(no_of_dataset):
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["#fe2f4a", "#8b88f8", "#fc824a", "lime",
                        "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def Plot_learning_per():
    eval = np.load('Eval_all_LP.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 3, 9]

    Algorithm = ['TERMS', 'RSA-WFS-HDRLDCN', 'EVO-WFS-HDRLDCN', 'GSOA-WFS-HDRLDCN', 'GOA-WFS-HDRLDCN', 'MGOA-WFS-HDRLDCN']
    Classifier = ['TERMS', 'Autoencoder', 'TCNN', 'LSTM', 'CapsNet', 'MGOA-WFS-HDRLDCN']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('--------------------------------------------------  75 % - learning percentage Dataset', i + 1,
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- 75 % - learning percentage Dataset', i + 1,
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
    for m in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[m, k, l, Graph_Term[j] + 4]

            length = np.arange(6)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='D', markerfacecolor='red',  # 98F5FF
                    markersize=6,  # 010fcc
                    label='RSA-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='s', markerfacecolor='green',  # 7FFF00
                    markersize=6,  # 08ff08
                    label='EVO-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='H', markerfacecolor='cyan',  # C1FFC1
                    markersize=8,  # fe420f
                    label='GSOA-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='p', markerfacecolor='#fdff38',
                    markersize=8,  # f504c9
                    label='GOA-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='w', markersize=12,
                    label='MGOA-WFS-HDRLDCN')

            plt.xticks(length, ('35', '45', '55', '65', '75', '85'))
            plt.xlabel('Learning Percentage')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path1 = "./Results/Dataset_%s_learn_per_%s_lrean.png" % (m + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(6)

            ax.bar(X + 0.00, Graph[:, 5], color='#f5054f', hatch='..', edgecolor='w', width=0.15,
                   label="Autoencoder")  # b
            ax.bar(X + 0.15, Graph[:, 6], color='#02ccfe', hatch='..', edgecolor='w', width=0.15,
                   label="TCNN")  # #ef4026
            ax.bar(X + 0.30, Graph[:, 7], color='#61e160', hatch='..', edgecolor='w', width=0.15,
                   label="LSTM")  # lime'
            ax.bar(X + 0.45, Graph[:, 8], color='#d725de', hatch='..', edgecolor='w', width=0.15,
                   label="CapsNet")  # y
            ax.bar(X + 0.60, Graph[:, 4], color='k', hatch='xx', edgecolor='w', width=0.15, label="MGOA-WFS-HDRLDCN")

            plt.xticks(X + 0.15, ('35', '45', '55', '65', '75', '85'))
            plt.xlabel('Learning Percentage')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path1 = "./Results/Dataset_%s_learn_per_%s_bar.png" % (m + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Plot_Batch_size():
    eval = np.load('Eval_all_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 3, 9]

    Algorithm = ['TERMS', 'RSA-WFS-HDRLDCN', 'EVO-WFS-HDRLDCN', 'GSOA-WFS-HDRLDCN', 'GOA-WFS-HDRLDCN',
                 'MGOA-WFS-HDRLDCN']
    Classifier = ['TERMS', 'Autoencoder', 'TCNN', 'LSTM', 'CapsNet', 'MGOA-WFS-HDRLDCN']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('--------------------------------------------------  128 - Batch size Dataset', i + 1,
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- 128 - Batch size Dataset', i + 1,
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)
    for m in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[m, k, l, Graph_Term[j] + 4]

            length = np.arange(5)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='D', markerfacecolor='red',  # 98F5FF
                    markersize=6,  # 010fcc
                    label='RSA-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='s', markerfacecolor='green',
                    # 7FFF00
                    markersize=6,  # 08ff08
                    label='EVO-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='H', markerfacecolor='cyan',  # C1FFC1
                    markersize=8,  # fe420f
                    label='GSOA-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='p', markerfacecolor='#fdff38',
                    markersize=8,  # f504c9
                    label='GOA-WFS-HDRLDCN')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='w', markersize=12,
                    label='MGOA-WFS-HDRLDCN')
            ax.fill_between(length, Graph[:, 0], Graph[:, 3], color='#acc2d9', alpha=.5)  # ff8400
            ax.fill_between(length, Graph[:, 3], Graph[:, 2], color='#c48efd', alpha=.5)  # 19abff
            ax.fill_between(length, Graph[:, 2], Graph[:, 1], color='#ff6f52', alpha=.5)  # 00f7ff
            ax.fill_between(length, Graph[:, 1], Graph[:, 4], color='#b2fba5', alpha=.5)  # ecfc5b
            plt.xticks(length, ('4', '16', '32', '64', '128'))
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path1 = "./Results/Dataset_%s_Batch_Size_%s_lrean.png" % (m + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#0165fc', edgecolor='w', width=0.15, label="Autoencoder")  # b
            ax.bar(X + 0.15, Graph[:, 6], color='#ff474c', edgecolor='w', width=0.15, label="TCNN")  # #ef4026
            ax.bar(X + 0.30, Graph[:, 7], color='#be03fd', edgecolor='w', width=0.15, label="LSTM")  # lime'
            ax.bar(X + 0.45, Graph[:, 8], color='#21fc0d', edgecolor='w', width=0.15, label="CapsNet")  # y
            ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="MGOA-WFS-HDRLDCN")
            plt.xticks(X + 0.15, ('4', '16', '32', '64', '128'))
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path1 = "./Results/Dataset_%s_Batch_Size_%s_bar.png" % (m + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def plot_Activation():
    eval = np.load('Eval_all_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 3, 9]
    Algorithm = ['TERMS', 'RSA-WFS-HDRLDCN', 'EVO-WFS-HDRLDCN', 'GSOA-WFS-HDRLDCN', 'GOA-WFS-HDRLDCN',
                 'MGOA-WFS-HDRLDCN']
    Classifier = ['TERMS', 'Autoencoder', 'TCNN', 'LSTM', 'CapsNet', 'MGOA-WFS-HDRLDCN']
    ACTIVATION = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Sigmoid - Activation Function Dataset', i + 1,
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Sigmoid - Activation Function Dataset', i + 1,
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2] + 1))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 11:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                    label="RSA-WFS-HDRLDCN")
            ax.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                    label="EVO-WFS-HDRLDCN")
            ax.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                    label="GSOA-WFS-HDRLDCN")
            ax.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                    label="GOA-WFS-HDRLDCN")
            ax.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='magenta',
                    markersize=12,
                    label="MGOA-WFS-HDRLDCN")
            plt.xticks(learnper, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.legend(loc=4)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path1 = "./Results/Dataset_%s_ACT_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="Autoencoder")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="TCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="LSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="CapsNet")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="MGOA-WFS-HDRLDCN")
            plt.xticks(X + 0.10, ('Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax'))
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path1 = "./Results/Dataset_%s_ACT_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    Plot_learning_per()
    Plot_Batch_size()
    plot_Activation()
