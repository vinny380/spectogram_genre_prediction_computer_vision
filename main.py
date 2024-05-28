# Importing necessary libraries from PyTorch for model building, optimization, and data manipulation
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import time
import copy
from subsampling import create_subsampled_dataset

# Setting a seed for reproducibility
np.random.seed(123)

# Customizing the ResNet50 model for a classification task with a specified number of classes
class ResNet50Custom(nn.Module):
    def __init__(self, num_classes=7):
        """Customized ResNet50 as given in section 4.4 of the report"""
        super(ResNet50Custom, self).__init__()
        # Loading the pre-trained ResNet50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Getting the number of features of the last layer
        num_ftrs = self.base_model.fc.in_features
        # Replacing the last fully connected layer with a new one that matches the number of classes
        # Adding dropout for regularization, as mentioned in section 4.3 of the report
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    # Forward pass definition
    def forward(self, x):
        return self.base_model(x)

# Function to evaluate the model's performance on a dataset
def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # Switching the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0
    total = 0

    # Iterating over batches of data in the specified DataLoader
    for inputs, labels in data_loader:
        inputs = inputs.to(device)  # Moving inputs to the device
        labels = labels.to(device)  # Moving labels to the device

        # Forward pass without gradient calculation
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Getting the predicted labels
            loss = criterion(outputs, labels)  # Calculating the loss

        # Accumulating the loss and the number of correct predictions
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    # Calculating average loss and accuracy
    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    # Printing loss and accuracy
    print(f'\nTest set: Average loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}\n')
    return epoch_loss, epoch_acc

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device, fold):
    """ Function to train the model, takes the instantiated model,
    loss function, optimizer, scheduler to lower learning rate,
    train and validations sets, number of epochs, device and current kth fold"""
    since = time.time()  # To measure the duration of the training
    best_model_wts = copy.deepcopy(model.state_dict())  # Copying the model's initial weights for checkpointing
    best_acc = 0.0  # Best accuracy initialization
    lowest_val_loss = float('inf')  # Lowest validation loss initialization

    # Looping through each epoch
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training and validation phase for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Setting the model to training mode
            else:
                model.eval()   # Setting the model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterating over data for the current phase
            data_loader = train_loader if phase == 'train' else val_loader
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass. Track gradients if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics accumulation
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Epoch statistics calculation
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Checkpointing the model if it has the lowest validation loss so far, as given in section 4.4 of the report
            if phase == 'val' and epoch_loss < lowest_val_loss:
                lowest_val_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Saving the model state dict with the best validation accuracy
                torch.save(model.state_dict(), f'model_best_val_fold_{fold}.pth')
                print(f"New best model saved at epoch {epoch+1} with loss {lowest_val_loss:.4f}, acc {best_acc:.4f}")

    # Printing the training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val Loss: {lowest_val_loss:.4f}, Best val Acc: {best_acc:.4f}')

    # Loading the best model weights
    model.load_state_dict(best_model_wts)
    return model

# The main function initializes and trains the model, also evaluates it on the test set.
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Setting the device
    num_epochs = 14  # Number of epochs to train
    k_folds = 5  # Number of folds for K-Fold Cross-Validation
    num_classes = 7  # Number of classes in the dataset

    # Defining transformations for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    source = 'data'  # Source directory of the dataset
    data_dir = 'subsampling'  # Directory where the undersampled dataset will be stored
    create_subsampled_dataset(source, data_dir, 1000)  # Function call to create a subsampled dataset with 1000 random samples from each class
    full_dataset = datasets.ImageFolder(data_dir)  # Loading the dataset

    # Splitting the dataset into training/validation and testing sets
    train_val_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=123)

    # Creating datasets for training/validation and testing phases with their respective transformations
    train_val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val'])

    # Creating DataLoaders for the train/validation and test subsets
    train_val_subset = Subset(train_val_dataset, train_val_indices)
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)

    # Performing K-Fold Cross-Validation
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_val_subset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Creating DataLoaders for each fold
        train_subsampler = Subset(train_val_subset, train_ids)
        val_subsampler = Subset(train_val_subset, val_ids)
        # Setting batch sizes to 32 as describe in section 4.5 of the report
        train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False, num_workers=4)

        # Model initialization
        model = ResNet50Custom(num_classes).to(device)
        # Stochastic Gradient Descent (SGD) optimizer as mention in section 4.4 of the report
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        # Learning rate scheduler for adaptive learning rate adjustments, as mention in section 4.4 of the report
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        # Cross-Entropy Loss with class weights to handle imbalanced datasets
        criterion = nn.CrossEntropyLoss()

        # Model training and validation for the current fold
        model_ft = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device, fold)

    # Final model evaluation on the test set
    print("\nFinal evaluation on the test set:")
    test_loss, test_acc = evaluate_model(model_ft, test_loader, criterion, device)

if __name__ == '__main__':
    main()
