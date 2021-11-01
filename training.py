from numpy.random.mtrand import shuffle
import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
import numpy as np
import pandas as pd
import time

def data_to_loader(train_dir,valid_dir):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            ])
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=bs,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=bs,shuffle=False)
    return trainloader,validloader

def model(device,unfreeze_n_last_layer):
    model = models.densenet161(pretrained=True)
    #  Freezing pretrained layers, so backpropogation doesn't occur during training.
    # for param in model.parameters():
    #     param.requires_grad = False
    if unfreeze_n_last_layer == 0:
        layer_count = 0
        for param in model.parameters():
            if layer_count <= 484-unfreeze_n_last_layer:
                param.requires_grad = False
            layer_count +=1
    model.classifier = nn.Sequential(nn.Linear(2208, 2208),
                                nn.ReLU(inplace=True),
                                nn.Dropout(do),
                                nn.Linear(2208, 1024),
                                nn.ReLU(inplace=True),
                                nn.Dropout(do),
                                nn.Linear(1024, 2),
                                nn.LogSoftmax(dim=1))
    model.to(device)
    print("model created")
    return model

def train(trainloader,validloader,total_epoch,criterion,device,root,model):
    print("Starting training")
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    train_losses,val_losses,accuracies = [], [], []
    for epoch in range(total_epoch):
        TP,FN,FP,TN = 0,0,0,0
        # train
        train_loss = 0
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            # optimise
            loss.backward()
            optimizer.step()
            # calculate loss
            train_loss += loss.item()


        # validate
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device),labels.to(device)
                outputs = model.forward(inputs)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                confusion_vector = top_class.flatten() / labels
                # - 1 and 1 (True Positive) 1
                # - 1 and 0 (False Positive) inf
                # - 0 and 0 (True Negative) nan
                # - 0 and 1 (False Negative) 0
                TP += torch.sum(confusion_vector == 1).item()
                FP += torch.sum(confusion_vector == float('inf')).item()
                TN += torch.sum(torch.isnan(confusion_vector)).item()
                FN += torch.sum(confusion_vector == 0).item()
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        assert (accuracy >= 0 and accuracy <= 1), "Accuracy must be greater than 0 and smaller than 1"
        train_losses.append(train_loss/len(trainloader))
        val_losses.append(val_loss/len(validloader))
        accuracies.append(accuracy)                 
        
        txt = "============= \n Epoch: {epoch_p:d} / {total_epoch:d} \t train_loss: {train_loss_p:.5f} \t val_loss: {val_loss_p:.5f}\t val_accuracy: {val_accuracy_p:.2f}\t ============="
        print(txt.format(epoch_p = epoch+1,
            total_epoch = total_epoch,
            train_loss_p = train_losses[-1],
            val_loss_p = val_losses[-1],
            val_accuracy_p = accuracies[-1],
        ))
        # save model for each epoch so that we can pick the best
        torch.save(model.state_dict(), os.path.join(root+"/models", 'epoch-{}.pth'.format(epoch)))
    return train_losses,val_losses,accuracies

## presets
root = os.getcwd()
# train_dir = root + "\\data\\tiny\\train"
# valid_dir = root + "\\data\\tiny\\val"

# Gideon's mac test
# root = "/Users/zhiiikaiii/Documents/GitHub/cancer/tiny"
# train_dir = root + "/train"
# valid_dir = root + "/val"

train_dir = root + "/data/Dataset120507/train"
valid_dir = root + "/data/Dataset120507/val"
test_dir = root + "/data/Dataset120507/test"

################################################################################
### Modify here with your configuration of hyper parameter
# Hyper-parameters
n_epochs = 5
bs = 2
do = 0.1
lr = 0.0001
unfreeze_n_last_layer = 200
################################################################################
total_epoch = n_epochs
criterion = nn.NLLLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric_csv = True # set metric_csv to be True to output metrics_date_time.csv file
if (model == 'resnet50_lsm'):
    assert (criterion == 'NLLLoss'), "only use resnet with logsoftmax with NLLLoss criterion"
if (model == 'resnet18_lsm'):
    assert (criterion == 'NLLLoss'), "only use resnet with logsoftmax with NLLLoss criterion"
assert (bs > 0), "Batch size cannot be < 1"
assert (lr >= 0 and lr < 1), "Learning rate must be 0 >= do < 1"
assert (do >= 0 and do <= 1), "Dropout rate must be 0 >= do < 1"
assert (total_epoch > 0), "Epochs must be a positive integer"
##start of script
torch.cuda.empty_cache() # clear GPU cache
trainloader,validloader = data_to_loader(train_dir,valid_dir)
model_1= model(device,unfreeze_n_last_layer)
train_losses,val_losses,accuracies = train(trainloader,validloader,total_epoch,criterion,device,root,model_1)

if (metric_csv):
    metric_output = '/metrics/training/'
    # df = pd.DataFrame([train_losses,val_losses,test_accuracies,test_sensitivities,test_specificities,test_precisions,test_f1scores]).T
    isExist = os.path.exists(root + metric_output)
    if not isExist:
        os.makedirs(root + metric_output)
    df = pd.DataFrame([train_losses,val_losses,accuracies]).T
    df.columns = ['train loss','val loss','val accuracies']
    df.insert(loc=0, column='epoch', value=np.arange(1,len(df)+1))
    now = time.strftime("%d%m%y_%H%M%S")
    df.to_csv(root + metric_output + '/metrics_' + now + ".csv", index = False)
print("finished training")
