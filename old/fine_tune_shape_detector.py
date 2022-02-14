from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import h5py
from PIL import Image


class FineTuneShapeDetector:
    def __init__(self):
        self.model = None
        self.data_dir = "/home/karthikm/active-perception/active-perception-project" \
                        "/shapes"

        self.num_classes = 4
        self.batch_size = 64
        self.num_epochs = 15
        self.image_datasets = None
        self.dataloaders_dict = None
        self.device = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()


    def setup_parameters(self):
        # Top level data directory. Here we assume the format of the directory conforms
        #   to the ImageFolder structure
        data_dir = "./data/hymenoptera_data"

        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        model_name = "squeezenet"

        # Number of classes in the dataset
        num_classes = 4

        # Batch size for training (change depending on how much memory you have)

        # Number of epochs to train for
        num_epochs = 15

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = True


    def train_model(self):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders_dict[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders_dict[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, val_acc_history

    def set_parameter_requires_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def setup_model(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v1', pretrained=True)
        self.set_parameter_requires_grad()
        self.model.classifier[1] = nn.Linear(1280, 4)
        print(self.model)

    def load_data(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in
                               ['train', 'val']}

        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x],
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                num_workers=4) for x in ['train', 'val']}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_optimizer(self):
        self.model = self.model.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        #params_to_update = self.model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
            #print("Num parameters to update: " + str(len(params_to_update)))

         # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


def main():
    fine_tune_detector = FineTuneShapeDetector()
    fine_tune_detector.setup_model()
    print(fine_tune_detector.model)
    fine_tune_detector.load_data()
    fine_tune_detector.create_optimizer()
    new_model, hist = fine_tune_detector.train_model()
    torch.save(new_model, "/home/karthikm/active-perception/active-perception-project/model/fine_tuned_model.pt")

    # model = torch.load("/home/karthikm/active-perception/active-perception-project/model/fine_tuned_model.pt")
    # model.eval()
    # for m in model.modules():
    #     for child in m.children():
    #         if type(child) == nn.BatchNorm2d:
    #             child.track_running_stats = False
    #             child.running_mean = None
    #             child.running_var = None
    #
    # model.eval()
    #
    # img = Image.open("/home/karthikm/active-perception/active-perception-project/shapes/random.jpg")
    # preprocess = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_tensor = preprocess(img)
    # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    #
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')
    #
    # with torch.no_grad():
    #     output = model(input_batch)
    #
    # print(output[0])
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)

if __name__ == "__main__":
    main()

