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
    print("SCRIPT RUNINNG")

    root = os.getcwd()
    datafolder = root + "/test_data"

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
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

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

        run_name = "{}-lr:{}-do:{}-bs:{}-opt:{}-crit:{}".format(config.model,config.lr,config.do,config.bs,config.optimizer,config.criterion)
        print("Running: " + run_name)
        wandb.run.name = run_name
        total_epoch = config.epochs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config.criterion == 'NLLLoss':       #do not use
            criterion = nn.NLLLoss()
        elif config.criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            assert False, "No such loss function."
        train_losses,val_losses,accuracies = [], [], []

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.bs,shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=config.bs,shuffle=False)
        
        if config.model == "resnet50":
            model = models.resnet50(pretrained=True)    # I think can't put in config
            #  Freezing pretrained layers, so backpropogation doesn't occur during training (only for resnet).
            for param in model.parameters():
                param.requires_grad = False
            # final layer of resnet18 has input for 2048 and output of 1000, we change this to 2 as we have 2 classes
            model.fc = nn.Sequential(nn.Linear(2048, 512),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(config.do),
                                            nn.Linear(512, 2),
                                            nn.LogSoftmax(dim=1))
        elif config.model =="resnet18":
            model = models.resnet18(pretrained=True)    # I think can't put in config
            #  Freezing pretrained layers, so backpropogation doesn't occur during training (only for resnet).
            for param in model.parameters():
                param.requires_grad = False
            # final layer of resnet50 has in_feat of 2048 and out of 1000, we change this to 2 as we have 2 classes
            model.fc = nn.Sequential(nn.Linear(512, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(config.do),
                                        nn.Linear(256, 2),
                                        nn.LogSoftmax(dim=1))
        else: 
            assert False, "NO SUCH MODEL"

        if config.optimizer =="Adam":
            optimizer = optim.Adam(model.fc.parameters(), lr=config.lr)

        model.to(device)
        wandb.watch(model, log="all")

        #start training
        print("entered sweep_train")
        # running_loss = 0
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1,verbose=True)
        for epoch in range(total_epoch):
            TP,FN,FP,TN = 0,0,0,0
            train_losses,val_losses,accuracies,sensitivities,specificities,precisions,f1scores=[],[],[],[],[],[],[]
            epoch_losses = []
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
                    for j in range(inputs.size()[0]):
                        if top_class[j] == 1 and labels[j] == 1:
                            TP +=1
                        elif top_class[j] == 1 and labels[j] == 0:
                            FP +=1
                        elif top_class[j] == 0 and labels[j] == 1:
                            FN +=1
                        elif top_class[j] == 0 and labels[j] == 0:
                            TN +=1



            scheduler.step()

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

            train_losses.append(running_loss/len(trainloader))
            val_losses.append(val_loss/len(validloader))
            # accuracies.append(accuracy/len(validloader))   
            accuracies.append(accuracy)   
            sensitivities.append(sensitivity) 
            specificities.append(specificity) 
            precisions.append(precision) 
            f1scores.append(F1score)                 
            
            epoch_losses.append(running_loss)
            # log epoch loss here
            print("Epoch:", epoch+1,"/",total_epoch)
            wandb.log({"Epoch": epoch, 
                        "train_loss": train_losses[-1],
                        "val_loss": val_losses[-1],
                        "val_accuracy": accuracies[-1],
                        "val_sensitivity":sensitivities [-1],
                        "val_specificity":specificities[-1],
                        "val_precision":precisions[-1],
                        "val_f1_score": f1scores[-1]
                        })
        print("finished training")
        return train_losses,val_losses,accuracies

import wandb
import os

# root= os.getcwd()
os.environ["WANDB_MODE"] = "dryrun" #Activate for offline testing

sweep_config = {
  "name" : "first-sweep",
  "method" : "grid",
  # "method": "bayes",
  'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'  
  },
  "parameters" : {
      "model" : {
            "value" : 'resnet18'
      },
      "epochs" : {
          "values" : [5]
      },
      "lr" : {
          "values" : [0.0001]
      },
      "do" : {
          "values" : [0.1]
      },
      "bs" : {
          "values" : [32]
      },
      "optimizer" : {
         "values" : ['Adam']
      },
      "criterion" : {
         "values" : ['CrossEntropyLoss']
      }
    }
}

wandb.login()
sweep_id = wandb.sweep(sweep_config, entity = 'team17', project = 'cancer')
max_runs = 10
wandb.agent(sweep_id, function=sweep, count=max_runs)
