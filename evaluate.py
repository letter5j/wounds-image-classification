# from model import model_128_all_512_7
from average_precision import mapk
import torch
import torch.nn.functional as F

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

from get_data import get_data_loaders


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          filename='confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    PATH = os.path.abspath(os.path.dirname(__file__))
    plt.savefig(os.path.join(PATH, 'result', filename))
    plt.close()


def create_confusion_matrix(y_true, y_pred, class_names, model_name):



    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix', filename=('%s_confusion_matrix.png' %(model_name)))


def cal_map(map_array):

    result = []

    for classification in map_array:
        result.append(sum(classification)/len(classification))
    return result

def eval_model(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################## generate result

    dataloaders, dataset_sizes, class_names = get_data_loaders()

    map_array = []
    for i in range(len(class_names)):
        map_array.append([])

    y_pred = []
    y_true = []

        # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        # m = torch.nn.Softmax(0)
        # XD = m(outputs)
        # print(outputs)
        # print(XD)
        _, preds = torch.max(outputs.data, 1)
        _, top_label = torch.topk(outputs.data, len(class_names))

        
        # print('predict ', preds)
        # print('True ', labels.data)
        y_pred += preds.cpu().numpy().flatten().tolist()
        y_true += labels.data.cpu().numpy().flatten().tolist()

        true_label = labels.data.cpu().numpy().flatten().tolist()
        top_label = top_label.cpu().numpy().tolist()
        # For mAP

        for i in range(len(top_label)):
            trueValue = true_label[i]
            probability = 0.0
            for j in range(len(top_label[i])):
                if(top_label[i][j] == trueValue):
                    probability = 1 / (j + 1)
            map_array[trueValue].append(probability)
        
        # print(mapk(labels.data.cpu().numpy().flatten().tolist(), preds.cpu().numpy().flatten().tolist(), k=10))
    
    create_confusion_matrix(y_true, y_pred, class_names, model.name)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(cal_map(map_array))
    



# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model_128_all_512_7.build_whole_model()

#     PATH = os.path.abspath(os.path.dirname(__file__))

#     path_to_model = os.path.join(PATH, 'pretrained_model')

#     model.load_state_dict(torch.load(os.path.join(path_to_model, '%s.pth' %(model.name))))
#     model.to(device)
#     model.train(False)

#     eval_model(model)