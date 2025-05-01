# Import general utilities like pandas, numpy, sklearn
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Logistic regression support with pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Define model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def run_logistic_regression(X_train, y_train, X_test, y_test,
                            n_epochs = 10, verbose=True):
    # Pytorch is much faster than sklearn for logistic regression
    # Convert to tensors and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    y_true = y_test_tensor.cpu().numpy()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # batch size is all data
    train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)

    model = LogisticRegression(input_dim=X_train.shape[1], num_classes=len(y_test)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch + 1}, loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    if verbose:
        print("----")
        print(f"Accuracy: {accuracy_score(y_true, preds):0.4f}")
        print(f"Macro F1: {f1_score(y_true, preds, average='macro'):0.4f}")
        print(f"Num correct: {sum(y_true == preds)}")

    return (accuracy_score(y_true, preds), f1_score(y_true, preds, average='macro'),
            sum(y_true == preds))