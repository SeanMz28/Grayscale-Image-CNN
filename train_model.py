import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

############################################################
# *NB: The Model was already trained, see "cnn_model.py"
# *This file is to show how the model was trained, Add "traindata.txt" and "trainlabels.txt" to folder
############################################################

# Clean up functions
def mean(arr):
    return sum(arr) / len(arr)

def contrasted(x, y, e):
    return abs(x - y) > e

def corner(origin, a, b, e):
    if contrasted(origin, a, e) and contrasted(origin, b, e):
        origin = mean([a, b])
    return origin

def side(origin, a, b, c, e):
    if contrasted(origin, a, e) and contrasted(origin, b, e) and contrasted(origin, c, e):
        origin = mean([a, b, c])
    return origin

def centre(origin, a, b, c, d, e):
    if contrasted(origin, a, e) and contrasted(origin, b, e) and contrasted(origin, c, e) and contrasted(origin, d, e):
        origin = mean([a, b, c, d])
    return origin

def clean(matrix, e):
    rowEnd = matrix.shape[0] - 1
    colEnd = matrix.shape[1] - 1

    matrix[0, 0] = corner(matrix[0, 0], matrix[0, 1], matrix[1, 0], e)
    matrix[0, colEnd] = corner(matrix[0, colEnd], matrix[0, colEnd - 1], matrix[1, colEnd], e)
    matrix[rowEnd, 0] = corner(matrix[rowEnd, 0], matrix[rowEnd - 1, 0], matrix[rowEnd, 1], e)
    matrix[rowEnd, colEnd] = corner(matrix[rowEnd, colEnd], matrix[rowEnd - 1, colEnd], matrix[rowEnd, colEnd - 1], e)

    for r in range(1, rowEnd):
        matrix[r, 0] = side(matrix[r, 0], matrix[r - 1, 0], matrix[r, 1], matrix[r + 1, 0], e)
    for r in range(1, rowEnd):
        matrix[r, colEnd] = side(matrix[r, colEnd], matrix[r - 1, colEnd], matrix[r, colEnd - 1], matrix[r + 1, colEnd], e)

    for c in range(1, colEnd):
        matrix[0, c] = side(matrix[0, c], matrix[0, c - 1], matrix[1, c], matrix[0, c + 1], e)
    for c in range(1, colEnd):
        matrix[rowEnd, c] = side(matrix[rowEnd, c], matrix[rowEnd, c - 1], matrix[rowEnd - 1, c], matrix[rowEnd, c + 1], e)

    for r in range(1, rowEnd):
        for c in range(1, colEnd):
            matrix[r, c] = centre(matrix[r, c], matrix[r, c - 1], matrix[r, c + 1], matrix[r - 1, c], matrix[r + 1, c], e)

    return matrix

def darken(matrix, e):
    row = matrix.shape[0]
    col = matrix.shape[1]
    for r in range(row):
        for c in range(col):
            if matrix[r, c] < e:
                matrix[r, c] = 10
    return matrix

def full_cleanup(images):
    for i in range(images.shape[0]):
        images[i, 0] = clean(images[i, 0], 160)
        images[i, 0] = darken(images[i, 0], 90)
        images[i, 0] = clean(images[i, 0], 50)
    return images

def preprocess_data(data):
    # Separate features and rotation values
    features = data[:, :-1]
    rotations = data[:, -1]

    # Identify columns with negative values (assumed noise)
    noise_columns = [col for col in range(features.shape[1]) if (features[:, col] < 0).any()]
    # Identify columns immediately after the noise columns
    adjusted_columns = [col + 1 for col in noise_columns if col + 1 < features.shape[1]]
    # Divide values in the adjusted columns by 10
    features[:, adjusted_columns] = features[:, adjusted_columns] / 10
    # Drop the noise columns
    features_cleaned = np.delete(features, noise_columns, axis=1)

    # Verify the new shape after removing noise columns
    print("Cleaned features shape:", features_cleaned.shape)

    # Check if we have 1024 columns after removing noise and rotation column
    if features_cleaned.shape[1] != 1024:
        raise ValueError("Unexpected number of columns after cleaning. Expected 1024 columns.")

    # Reshape each row into a 32x32 image and add channel dimension
    images = features_cleaned.reshape(-1, 32, 32, 1).transpose(0, 3, 1, 2)

    images = full_cleanup(images)

    for i in range(len(images)):
        if rotations[i] == 2:
            images[i] = np.rot90(images[i], 2, axes=(1, 2))
        elif rotations[i] == 1:
            images[i] = np.rot90(images[i], 3, axes=(1, 2))
        elif rotations[i] == 3:
            images[i] = np.rot90(images[i], 1, axes=(1, 2))

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0
    return images

def load_data():
    # Load the dataset using np.genfromtxt
    data = np.genfromtxt('traindata.txt', dtype=float, delimiter=',')
    labels = np.genfromtxt('trainlabels.txt', dtype=int, delimiter=',')

    images = preprocess_data(data)
    # One-hot encode labels
    labels = np.eye(21)[labels]

    # Split data into training and validation set
    train_images, train_labels = images[:4200], labels[:4200]
    val_images, val_labels = images[4200:], labels[4200:]

    return train_images, train_labels, val_images, val_labels

def create_data_loaders(train_images, train_labels, val_images, val_labels):
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_images = torch.tensor(val_images, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.dropout = nn.Dropout(0.07)
        self.fc2 = nn.Linear(64, 21)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_and_evaluate(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train = 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == torch.max(labels, 1)[1]).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_loader.dataset)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        model.eval()
        val_loss, correct_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, torch.max(labels, 1)[1])

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == torch.max(labels, 1)[1]).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct_val / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def main():
    train_images, train_labels, val_images, val_labels = load_data()
    train_loader, val_loader = create_data_loaders(train_images, train_labels, val_images, val_labels)

    model = CNNModel()
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_and_evaluate(model, train_loader, val_loader)

    #uncomment to save the trained model
    #save_model(model, 'cnn_model.pth')

if __name__ == "__main__":
    main()
