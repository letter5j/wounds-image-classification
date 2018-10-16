import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import visdom

from get_data import get_data_loaders

import numpy as np
import time
import copy
import os

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):

    # By this way you should close the file directly:
    PATH = os.path.abspath(os.path.dirname(__file__))
    result_file = open(os.path.join(PATH, 'result', '%s.txt' %(model.name)), 'w+')
    # result_file.writelines(text)
    # result_file.close()
    # By this way the file closed after the indented block after the with has finished execution:

    # with open("filename.txt", "w") as fh:
    #     fh.write(text)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    viz = visdom.Visdom()
    #线图用来观察loss 和 accuracy
    accuracy_line = viz.line(X=np.arange(1, 10, 1), Y=np.arange(1,10,1))
    #线图用来观察loss 和 accuracy
    loss_line = viz.line(X=np.arange(1, 10, 1), Y=np.arange(1,10,1))
    #text 窗口用来显示loss 、accuracy 、时间
    # text = viz.text("FOR TEST")


    time_point, loss_point, accuracy_point = [], [], []
    time_point_t, loss_point_t, accuracy_point_t = [], [], []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        result_file.writelines('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        result_file.writelines('-' * 10 + '\n')


        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            result_file.writelines('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))


            if phase == 'train':
                time_point.append(epoch)
                loss_point.append(epoch_loss)
                accuracy_point.append(epoch_acc)
            if phase == 'test':
                time_point_t.append(epoch)
                loss_point_t.append(epoch_loss)
                accuracy_point_t.append(epoch_acc)                

                # viz.text("<h3 align='center' style='color:blue'>accuracy : {}</h3><br><h3 align='center' style='color:pink'>"
                #         "loss : {:.4f}</h3><br><h3 align ='center' style='color:green'>time : {:.1f}</h3>"
                #         .format(epoch_acc,epoch_loss,time.time()-since),win =text)
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        viz.line(X=np.column_stack((np.array(time_point),np.array(time_point_t))),
                Y=np.column_stack((np.array(accuracy_point),np.array(accuracy_point_t))),
                win=accuracy_line,
                opts=dict(title='%s-ac' %(model.name), legend=["accuracy", "test_accuracy"]))
        viz.line(X=np.column_stack((np.array(time_point),np.array(time_point_t))),
                Y=np.column_stack((np.array(loss_point),np.array(loss_point_t))),
                win=loss_line,
                opts=dict(title='%s-loss' %(model.name), legend=["loss", "test_loss"]))

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    result_file.writelines('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    result_file.writelines('Best val Acc: {:4f}\n'.format(best_acc))

    # file close
    result_file.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def start_train(model):
    dataloaders, dataset_sizes, class_names = get_data_loaders()
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.Adam(model.classifier.parameters(), lr=0.01)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.01)

    model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, 20, dataloaders, dataset_sizes)

    PATH = os.path.abspath(os.path.dirname(__file__))
    
    torch.save(model.state_dict(), os.path.join(PATH, 'pretrained_model', '%s.pth' %(model.name)))
    return model