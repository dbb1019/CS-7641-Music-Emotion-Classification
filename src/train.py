import torch
import torch.nn as nn
import torch.optim as optim
from utils import preprocess_data, calculate_metrics, preprocess_data_dynamic
from dataset import EmotionDataset
from torch.utils.data import DataLoader, random_split
from models import EmotionCNN, FCN
from sklearn.metrics import r2_score
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error

torch.manual_seed(1)
# Prepare the data
song_level = True
if song_level:
    X_train, X_test, y_train, y_test = preprocess_data(test_size=0.2, random_state=1234)
    X_train = X_train.reshape(-1, 1, 260)
    X_test = X_test.reshape(-1, 1, 260)
else:
    X_train, X_test, y_train, y_test = preprocess_data_dynamic(
        test_size=0.2, random_state=1234
    )
    X_train = np.transpose(X_train, axes=(0, 2, 1))
    X_test = np.transpose(X_test, axes=(0, 2, 1))
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)


train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

# Train the model
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()
    # Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}"
    )

# Test the model
model.eval()
test_loss = 0
test_outputs = []
test_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        test_outputs.extend(outputs.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

print(f"Test Loss: {test_loss / len(test_loader):.4f}")

test_outputs = np.array(test_outputs)
test_targets = np.array(test_targets)
print("test_outputs:", test_outputs.shape)
print("test_targets:", test_targets.shape)

# r2_arousal = r2_score(test_targets[:, 0], test_outputs[:, 0])
# r2_valence = r2_score(test_targets[:, 1], test_outputs[:, 1])
if song_level:
    (
        emotion_accuracy,
        emotion_precision,
        emotion_recall,
        emotion_f1_score,
        subemotion_accuracy,
        subemotion_precision,
        subemotion_recall,
        subemotion_f1_score,
        r2,
        mse,
    ) = calculate_metrics(test_outputs, test_targets)
else:
    r2_arousal = r2_score(test_targets[:, :60], test_outputs[:, :60])
    r2_valence = r2_score(test_targets[:, -60:], test_outputs[:, -60:])
    print(f"R2 Score - Arousal: {r2_arousal:.4f}")
    print(f"R2 Score - Valence: {r2_valence:.4f}")
    print("test_outputs[0]", test_outputs[0])
    print("test_targets[0]", test_targets[0])
    rmse_arousal = np.sqrt(
        mean_squared_error(test_targets[:, :60], test_outputs[:, :60])
    )
    rmse_valence = np.sqrt(
        mean_squared_error(test_targets[:, -60:], test_outputs[:, -60:])
    )
    print("RMSE Arousal:", rmse_arousal)
    print("RMSE Valence:", rmse_valence)
