import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

# Dataset Preparation
dataset = pd.read_csv("facial_expressions/data/legend.csv")
dataset = dataset.drop(["user.id"], axis=1)
dataset['emotion'] = dataset['emotion'].str.lower()

class_mapping = dict(zip(dataset["emotion"].astype('category').cat.codes, dataset["emotion"]))
dataset["emotion_class"] = dataset["emotion"].astype('category').cat.codes

xtrain, xval, ytrain, yval = train_test_split(
    dataset["image"], dataset["emotion_class"], test_size=0.2, random_state=77, stratify=dataset["emotion_class"]
)
weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset["emotion_class"]), y=dataset["emotion_class"])

# Dataset Class
class EmotionTrainDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open("/home/dikshans/experiments/facial_expressions/images/Aaron_Eckhart_0001.jpg").convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)) / 255.
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.labels.iloc[idx].astype(np.float32)
        return image, label

# Ray Tune Model
class CNNModel(nn.Module):
    def __init__(self, input_channels=3, conv_layers=None, fc_layers=None, num_classes=8, dropout=0.5):
        super(CNNModel, self).__init__()
        if conv_layers is None:
            conv_layers = [8, 16, 32]
        if fc_layers is None:
            fc_layers = [128]

        self.features = nn.Sequential()
        in_channels = input_channels

        for i, out_channels in enumerate(conv_layers):
            self.features.add_module(f"conv{i+1}", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.features.add_module(f"relu{i+1}", nn.LeakyReLU())
            self.features.add_module(f"pool{i+1}", nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        feature_size = 128 // (2 ** len(conv_layers))
        flattened_size = conv_layers[-1] * (feature_size ** 2)

        self.classifier = nn.Sequential()
        in_features = flattened_size
        for i, out_features in enumerate(fc_layers):
            self.classifier.add_module(f"fc{i+1}", nn.Linear(in_features, out_features))
            self.classifier.add_module(f"relu_fc{i+1}", nn.LeakyReLU())
            self.classifier.add_module(f"dropout_fc{i+1}", nn.Dropout(dropout))
            in_features = out_features

        self.classifier.add_module("fc_out", nn.Linear(in_features, num_classes))
        self.classifier.add_module("softmax", nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training Function
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    traindataset = EmotionTrainDataset(xtrain, ytrain)
    valdataset = EmotionTrainDataset(xval, yval)
    train_loader = DataLoader(traindataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(valdataset, batch_size=512, shuffle=True)

    # Define model
    model = CNNModel(
        conv_layers=config["conv_layers"],
        fc_layers=config["fc_layers"],
        dropout=config["dropout"],
        num_classes=len(class_mapping)
    ).to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    best_val_acc = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Report metrics to Ray Tune
        train.report({"loss": val_loss / len(val_loader), "accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc

# Ray Tune Configuration
search_space = {
    "conv_layers": tune.choice([[16, 32, 16], [16, 32, 64], [16, 32, 38], [8, 16, 32], [32, 64, 128]]),
    "fc_layers": tune.choice([[32], [64], [32, 64], [128], [512], [256, 128], [512, 256, 128]]),
    "dropout": tune.uniform(0.3, 0.7),
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "epochs": 50
}

# Run Ray Tune
ray.init(ignore_reinit_error=True)
scheduler = ASHAScheduler(metric="accuracy", mode="max")
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=100,  # Number of trials
    scheduler=scheduler,
    resources_per_trial={"cpu": 16, "gpu": 4}  # Adjust based on your hardware
)

# Best Configuration
print("Best hyperparameters found were: ", analysis.get_best_trial("accuracy", "max"))
