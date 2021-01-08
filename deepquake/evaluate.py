import glob
import argparse
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from models import load_model, load_dataset, CLASSES_BINS
import local_config


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', help='file with saved weights')
    parser.add_argument('--feature',
                        choices=['detection', 'distance', 'magnitude', 'azimuth'],
                        required=True,
                        help='which feature to evaluate')
    args = parser.parse_args()

    streamsPathEvent = local_config.PATH_TEST_EVENT

    files = glob.glob(streamsPathEvent + "/*.tfrecords")

    if args.feature == 'detection':
        streamsPathNoise = local_config.PATH_TEST_NOISE
        files2 = glob.glob(streamsPathNoise + "/*.tfrecords")
        files = files + files2

    # Creating the NN model
    model = load_model(args.feature)
    print(model.summary())

    model.load_weights(args.weights)

    bs = 64000
    eval_dataset = load_dataset(files, args.feature).batch(bs)

    loss, acc = model.evaluate(eval_dataset, verbose=2)

    print('Loss:', loss)
    print('Accuracy:', acc)

    # Predicted azimuth classes
    predictions = model.predict(eval_dataset)
    pred_classes = [np.argmax(x) for x in predictions]

    # Target classes
    targets = eval_dataset.map(lambda data, t: t)
    true_values = tf.data.experimental.get_single_element(targets)
    print(true_values)

    ax = plot_confusion_matrix(true_values, pred_classes, CLASSES_BINS[args.feature][:],
                               title=args.feature)
    plt.show()
    #creating a classification report with precision and recall and f-1 score (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
    clasRep = classification_report(true_values, pred_classes)
    print(clasRep)
