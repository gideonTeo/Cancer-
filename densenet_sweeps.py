from PIL.Image import new
from numpy.random.mtrand import shuffle
import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
import numpy as np
import pandas as pd
import matplotlib
import wandb
import sklearn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
# import splitfolders 

# import load_split_train_test
def sweep(config=None):
    bs = 32
    print("SCRIPT RUNINNG")
    root = os.getcwd()
    datafolder = root + "/data/Dataset120507"
    print(datafolder)

    # Gideon's mac testing
    # datafolder = "/Users/zhiiikaiii/Documents/GitHub/cancer/tiny"

    train_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            ])
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        ])
    train_dir = datafolder + "/train"
    valid_dir = datafolder + "/val"
    # train_dir = datafolder + "/train"
    # valid_dir = datafolder + "/val"
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=bs,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=bs,shuffle=False)

    test_dir = datafolder + "/test"
    test_transforms = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=bs,shuffle=True)
    
    # initialise a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        #### ASSERT ANY INVALID COMBINATION/PARAMETERS####
        if (config.model == 'resnet50_lsm'):
            assert (config.criterion == 'NLLLoss'), "only use resnet with logsoftmax with NLLLoss criterion"
        if (config.model == 'resnet18_lsm'):
            assert (config.criterion == 'NLLLoss'), "only use resnet with logsoftmax with NLLLoss criterion"
        assert (config.bs > 0), "Batch size cannot be < 1"
        assert (config.lr >= 0 and config.lr < 1), "Learning rate must be 0 >= do < 1"
        assert (config.do >= 0 and config.do <= 1), "Dropout rate must be 0 >= do < 1"
        assert (config.epochs > 0), "Epoch must be a positive integer"

        #######################################
        run_name = "{}-unfreeze_n_layer:{}-do:{}-w_decay:{}-opt:{}-crit:{}".format(config.model,config.unfreeze_n_last_layer, config.do,config.w_decay,config.optimizer,config.criterion)
        print("Running: " + run_name)
        wandb.run.name = run_name
        total_epoch = config.epochs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #clear GPU cache
        torch.cuda.empty_cache()
        if config.criterion == 'NLLLoss':       
            criterion = nn.NLLLoss()
        elif config.criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
            # Binary Cross-Entropy Loss
            # Hinge Loss
            # Squared Hinge Loss
            # SigmoidCrossEntropyLoss
        else:
            assert False, "No such loss function."           
        if config.model =="densenet161":
            model = models.densenet161(pretrained=True)
            if config.unfreeze_n_last_layer == 0:
                layer_count = 0
                for param in model.parameters():
                    if layer_count <= 484-config.unfreeze_n_last_layer:
                        param.requires_grad = False
                    layer_count +=1
            model.classifier = nn.Sequential(nn.Linear(2208, 2208),
                            nn.ReLU(inplace=True),
                            nn.Dropout(config.do),
                            nn.Linear(2208, 1024),
                            nn.ReLU(inplace=True),
                            nn.Dropout(config.do),
                            nn.Linear(1024, 2),
                            nn.LogSoftmax(dim=1))
        if config.optimizer =="Adam":
            optimizer = optim.Adam(model.classifier.parameters(), lr=config.lr, weight_decay= config.w_decay)

        model.to(device)
        wandb.watch(model, log="all")

        #start training
        print("entered sweep_train")
        # running_loss = 0
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1,verbose=True)
        for epoch in range(total_epoch):
            TP,FN,FP,TN = 0,0,0,0
            train_losses,val_losses,accuracies=[],[],[]
            test_accuracies,test_sensitivities,test_specificities,test_precisions,test_f1scores = [],[],[],[],[]
            running_loss = 0
            # train
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
                running_loss += loss.item()
            # validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(validloader):
                    inputs, labels = inputs.to(device),labels.to(device)
                    outputs = model.forward(inputs)
                    batch_loss = criterion(outputs, labels)
                    val_loss += batch_loss.item()
                    
                    ps = torch.exp(outputs)
                    #take top probability
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
            # save model for each epoch so that we can pick the best
            # torch.save(model.state_dict(), os.path.join(root+"//models//"+run_name, 'epoch-{}.pth'.format(epoch)))
            scheduler.step()
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            # sensitivity = TP/(TP+FN)
            # specificity = TN/(TN+FP)
            # precision = TP/(TP+FP)
            # F1score = TP/(TP+0.5*(FP+FN))

            train_losses.append(running_loss/len(trainloader))
            val_losses.append(val_loss/len(validloader))
            accuracies.append(accuracy)   
            # sensitivities.append(sensitivity) 
            # specificities.append(specificity) 
            # precisions.append(precision) 
            # f1scores.append(F1score)                 
            # log epoch loss here
            print("Epoch:", epoch+1,"/",total_epoch)
            wandb.log({"Epoch": epoch, 
                        "train_loss": train_losses[-1],
                        "val_loss": val_losses[-1],
                        "val_accuracy": accuracies[-1]
                        # "val_sensitivity":sensitivities [-1],
                        # "val_specificity":specificities[-1],
                        # "val_precision":precisions[-1],
                        # "val_f1_score": f1scores[-1]
                        },commit = False)
            print("finished epoch training")
            # test
            TP,FN,FP,TN = 0,0,0,0 # reset for test data
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                confusion_vector = top_class.flatten() / labels

                TP += torch.sum(confusion_vector == 1).item()
                FP += torch.sum(confusion_vector == float('inf')).item()
                TN += torch.sum(torch.isnan(confusion_vector)).item()
                FN += torch.sum(confusion_vector == 0).item()
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

            test_accuracies.append(accuracy)
            test_sensitivities.append(sensitivity) 
            test_specificities.append(specificity) 
            test_precisions.append(precision) 
            test_f1scores.append(F1score) 
            wandb.log({"test_accuracy": test_accuracies[-1],
                        "test_sensitivity":test_sensitivities [-1],
                        "test_specificity":test_specificities[-1],
                        "test_precision":test_precisions[-1],
                        "test_f1_score": test_f1scores[-1]
                        })
        # sweep finished

import wandb
import os


################################################################################
### Modify here for changing hyper-parameter options for tuning
# root= os.getcwd()
# comment out to allow sending sweep information to online server for visualisation
# os.environ["WANDB_MODE"] = "dryrun" #Activate for offline testing

sweep_config = {
  "name" : "densenet-sweep",
  "method": "grid",
  'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'  
  },
  # modify the list of options for parameters below
  "parameters" : {
      "model" : {
            "value" : 'densenet161'
      },
      "epochs" : {
          "values" : [15]
      },
      "lr" : {
          "values" : [0.0001]
      },
      "do" : {
          "values" : [0]
      },
      "bs" : {
          "values" : [32]
      },
      "optimizer" : {
         "values" : ['Adam']
      },
      "criterion" : {
         "values" : ['NLLLoss']
      },
      "w_decay":{
          "values" : [0.001]
      },
      "unfreeze_n_last_layer":{
          "values" : [0,5,10,20]
      }
    }
}
# maximum number of configurations to run
max_runs = 10

wandb.login()
# change to your entity and project when running
sweep_id = wandb.sweep(sweep_config, entity = 'team17', project = 'cancer')

##################################################################################
wandb.agent(sweep_id, function=sweep, count=max_runs)
