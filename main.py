import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

# Load and preprocess the dataset
data = pd.read_csv("events.csv")
data = data[data["event"] == "view"]
data["session_id"] = data["visitorid"].astype("category").cat.codes

# Encode item ids
item_encoder = LabelEncoder()
data["itemid"] = item_encoder.fit_transform(data["itemid"])

# Split dataset into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the Dataset class
class SessionDataset(Dataset):
    def __init__(self, data, session_length=5):
        self.data = data
        self.session_length = session_length

    def __len__(self):
        return len(self.data) - self.session_length

    def __getitem__(self, idx):
        X = self.data.iloc[idx : idx + self.session_length]["itemid"].values
        y = self.data.iloc[idx + self.session_length]["itemid"]
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Create the LSTM model
class SessionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SessionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, _ = self.lstm(embedded.view(input.size(0), -1, self.hidden_size))
        output = self.out(output[:, -1, :])
        return output


# Create DataLoader objects
train_dataset = SessionDataset(train_data)
test_dataset = SessionDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

# Initialize model, loss function, and optimizer
n_items = len(item_encoder.classes_)
model = SessionLSTM(n_items, 128, n_items)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
total_loss = 0
for epoch in range(num_epochs):
    start_time = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}]')

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, target in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Accuracy of the model on the test data: {100 * correct / total}%")


# Function to recommend the next item
def recommend_next_item(session):
    session_encoded = torch.tensor([item_encoder.transform(session)], dtype=torch.long)
    with torch.no_grad():
        output = model(session_encoded)
        predicted = torch.argmax(output).item()
        recommended_item = item_encoder.inverse_transform([predicted])[0]
        return recommended_item

# Example: Get a user input and recommend the next item
user_session = []
print("Enter 5 item ids, one at a time:")
for i in range(5):
    item_id = int(input(f"Item {i + 1}: "))
    user_session.append(item_id)

recommended_item = recommend_next_item(user_session)
print(f"Recommended next item: {recommended_item}")
