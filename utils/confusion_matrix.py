import numpy as np
import argparse


def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        #cmap = plt.get_cmap('Pastel1')

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #clb = plt.colorbar()
    #clb.set_label('Number of samples')
    #clb.ax.set_title('# of samples', size=10)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, size=font_size)  #, rotation=45
        plt.yticks(tick_marks, target_names, size=font_size)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title('CM, Normalized - Accuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass), size=16)
    else:
        plt.title('CM - Accuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass), size=16)

    import matplotlib.font_manager as font_manager
    font_prop = font_manager.FontProperties(size=20)
    thresh = np.nanmax(cm) / 1.5 if normalize else np.nanmax(cm) / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:2.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontproperties=font_prop)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=16)
    plt.xlabel('\nPredicted label', size=16)
    plt.savefig('CM_U.png', bbox_inches='tight', dpi=100)
    plt.show()


if __name__ == "__main__":
    # Parameters
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("is_normalized", help="a bool param")
    arg_parser.add_argument("font_size", help="int font size")
    args = arg_parser.parse_args()

    # Parse arguments
    is_normalized = args.is_normalized
    font_size = args.font_size

    # paper data
    cm1 = np.array([[95313, 156, 517, 1279, 1542],
                                        [6179, 123631, 468, 19, 264],
                                        [349, 21, 472004, 47, 58],
                                        [68, 2, 567, 71075, 77],
                                        [6372, 269, 4498, 42, 462915]])

    cm2 = np.array([[30, 28 ],
                          [22, 40]])

    plot_confusion_matrix(cm=cm2,
                          normalize=is_normalized,
                          target_names=['Fake', 'Real'])
