import os
import torch 
from torch import nn
from torchvision import datasets, transforms, models
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import time

root = os.getcwd()

# Gid's
# root = "/Users/zhiiikaiii/Documents/GitHub/cancer"

metric_csv = True
def test(test_dir,bs,do,pth_path, verbose = False):
    test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor()])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=bs,shuffle=True)
    model = models.densenet161(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(2208, 2208),
                            nn.ReLU(inplace=True),
                            nn.Dropout(do),
                            nn.Linear(2208, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(do),
                            nn.Linear(1024, 2),
                            nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(pth_path))
    model.to(device)
    model.eval()

    TP,FN,FP,TN = 0,0,0,0
    count = 0
    start = timer()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
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
        if verbose:
            if count%1000 == 0:
                end = timer()
                print(count)
                print("time: ", (end - start)/60, "minutes")
                start = end
        count +=1
    return TP,TN,FP,FN
    

def metrics_calulation (TP,TN,FP,FN):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    F1score = TP/(TP+0.5*(FP+FN))
    assert (accuracy >= 0 and accuracy <= 1), "Accuracy must be greater than 0 and smaller than 1"
    assert (sensitivity >= 0 and sensitivity <= 1), "Sensitivity must be greater than 0 and smaller than 1"
    assert (specificity >= 0 and specificity <= 1), "Specificity must be greater than 0 and smaller than 1"
    assert (precision >= 0 and precision <= 1), "Precision must be greater than 0 and smaller than 1"
    assert (F1score >= 0 and F1score <= 1), "F1-score must be greater than 0 and smaller than 1"

    return accuracy, sensitivity, specificity, precision, F1score

test_dir = root + "/../data/Dataset120507/test"

#Gideon's mac testing
# test_dir= "/Users/zhiiikaiii/Documents/GitHub/cancer/tiny/test"

model_names = os.listdir(root+ "/../models")
accuracies,sensitivities,specificities,precisions,F1scores = [],[],[],[],[]
for pth in model_names:
    if pth[-4:] == ".pth":
        print("testing : "+ pth)
        bs = 1
        do = 0.1
        assert (bs > 0), "Batch size cannot be < 1"
        assert (do >= 0 and do <= 1), "Dropout rate must be 0 >= do < 1"
        pth_path = root + "/../models/"+pth
        TP,TN,FP,FN = test(test_dir,bs,do,pth_path,verbose= False)
        accuracy, sensitivity, specificity, precision, F1score = metrics_calulation(TP,TN,FP,FN)
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        F1scores.append(F1score)

# cerate a "metrics" folder in root directory and within it "testing"
if (metric_csv):
    metric_output = '/metrics/testing/'
    isExist = os.path.exists(root + metric_output)
    if not isExist:
        os.makedirs(root + metric_output)
    df = pd.DataFrame([accuracies,sensitivities,specificities,precisions,F1scores]).T
    df.columns = ["accuracies","sensitivities","specificities","precisions","F1scores"]
    df.insert(loc=0, column='epoch', value=np.arange(1,len(df)+1))
    now = time.strftime("%d%m%y_%H%M%S")
    df.to_csv(root + metric_output + 'metrics_' + now + ".csv", index = False)
print("finished training")

