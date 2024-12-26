import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import sklearn
# import tensorflow as tf
# import matplotlib
# import seaborn as sns


##################################################################################################################
# *NB: The MODEL was already trained, see "cnn_model.pth" in the folder, and the model is loaded in the main function.
# *You can see how the model was trained in "train_model.py"
###################################################################################################################

# Clean up functions to smooth out any remaining noise.
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
    # Check if we have 1024 columns after removing noise and rotation column
    if features_cleaned.shape[1] != 1024:
        raise ValueError("Unexpected number of columns after cleaning. Expected 1024 columns.")

    # Rotate images based on rotation values
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

def main():
    # Load the pre-trained model
    model = CNNModel()
    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval()

    # Load test data
    test_data = np.genfromtxt('testdata.txt', dtype=float, delimiter=',')
    test_images = preprocess_data(test_data)

    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_dataset = TensorDataset(test_images)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Classify
    infer_labels = []
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images[0])
            _, predicted = torch.max(outputs, 1)
            infer_labels.extend(predicted.numpy())

    infer_labels = pd.DataFrame(infer_labels)

    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (test_data.shape[0], 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(test_data.shape[0], 1)}"

    infer_labels.to_csv("predlabels.txt", index=False, header=False)

if __name__ == "__main__":
    main()
