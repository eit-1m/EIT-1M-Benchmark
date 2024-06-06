import torch
import random
import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, mobilenet_v2, resnet34, resnet50
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from dataset import NPYDataset
import argparse
from tqdm import tqdm
from torcheeg.models import EEGNet
from torcheeg import transforms

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_path', default='', help='Please add your model save directory')
    parser.add_argument('--model', default='resnet18', help='Please add your model name')
    parser.add_argument('--lr', default=0.001, type=float, help='Please add your learning rate')
    parser.add_argument('--modality', default='image', help='Please add your modality')
    parser.add_argument('--datasets', default='data_0528', help='Please add your datasets')
    parser.add_argument('--data_root', default='data/', help='Please add your datasets')

    args = parser.parse_args()
    print(args)
    save_path = args.save_path

    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)  

    num_classes = 10
    num_channels = 63
    num_length = 101
    print("Model: ", args.model)
    if args.model == 'resnet18':
        model = resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        transform = transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor(),
        ])
    elif args.model == 'resnet34':
        model = resnet34(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        transform = transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor(),
        ])
    elif args.model == 'resnet50':
        model = resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        transform = transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor(),
        ])
    elif args.model == 'eegnet':
        model = EEGNet(chunk_size=num_length,
               num_electrodes=num_channels,
               dropout=0.25,
               kernel_1=256,
               kernel_2=64,
               F1=32,
               F2=64,
               D=8,
               num_classes=num_classes)
        transform = transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor(),
        ])
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        transform = transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor(),
        ])
        
    print(model)


    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.8)


    # Load the dataset
    dataset_list = args.datasets.split('+')
    train_path = [args.data_root+dataset+'/train' for dataset in dataset_list]

    print("modality: ", args.modality)
    train_dataset = NPYDataset(root_dirs=train_path, transform=transform, modality=args.modality, padding=padding)
    print("Train dataset size: ", len(train_dataset))   

    # Split train dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

    # Create data loaders
    batch_size = 2048
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,num_workers=16)

    # Training and evaluation loop
    num_epochs = 500

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for data, labels in tqdm(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        train_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2%}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []

        if epoch % 1 == 0 or epoch == num_epochs-1:
            
            with torch.no_grad():
                for data, labels in tqdm(val_loader):
                    data = data.to(device)
                    labels = labels.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
            val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            print(f'Validation Accuracy after Epoch {epoch+1}: {val_accuracy:.2%}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')
            # print(confusion_matrix(all_labels, all_preds))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior

if __name__ == "__main__":    
    setup_seed(3407)
    main()
