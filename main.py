import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

# Load and preprocess the dataset
data = pd.read_csv("retailrocket_50percent.csv")
data = data[data["event"] == "view"]
data["session_id"] = data["visitorid"].astype("category").cat.codes

# Encode item ids and item types
item_encoder = LabelEncoder()
data["itemid"] = item_encoder.fit_transform(data["itemid"])
type_encoder = LabelEncoder()
data["item_type"] = type_encoder.fit_transform(data["item_type"])

# Define the Dataset class
class SessionDataset(Dataset):
    def __init__(self, data, session_length=5):
        self.data = data
        self.session_length = session_length

    def __len__(self):
        return len(self.data) - self.session_length

    def __getitem__(self, idx):
        X = self.data.iloc[idx : idx + self.session_length][["itemid", "item_type"]].values
        y = self.data.iloc[idx + self.session_length]["itemid"]
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Create the LSTM model
class SessionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SessionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_item = nn.Embedding(input_size, hidden_size)
        self.embedding_type = nn.Embedding(len(type_encoder.classes_), hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded_item = self.embedding_item(input[:, :, 0])
        embedded_type = self.embedding_type(input[:, :, 1])
        embedded = torch.cat((embedded_item, embedded_type), -1)
        output, _ = self.lstm(embedded)
        output = self.out(output[:, -1, :])
        return output

# Use a smaller sample of the dataset (because it takes too much time to load the full dataset)
sample_fraction = 0.01
data_sample = data.sample(frac=sample_fraction, random_state=42)

# Split the sampled dataset into train and test
train_data_sample, test_data_sample = train_test_split(data_sample, test_size=0.2, random_state=42)

# Create DataLoader objects with the smaller dataset
train_dataset_sample = SessionDataset(train_data_sample)
test_dataset_sample = SessionDataset(test_data_sample)
train_loader_sample = DataLoader(train_dataset_sample, batch_size=250, shuffle=True)
test_loader_sample = DataLoader(test_dataset_sample, batch_size=250, shuffle=False)

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
    for i, (inputs, target) in enumerate(train_loader_sample):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader_sample)}]')

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, target in test_loader_sample:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Accuracy of the model on the test data: {100 * correct / total}%")


# Function to recommend the next item
def recommend_next_item(session, session_types):
    session_encoded = torch.tensor([item_encoder.transform(session)], dtype=torch.long)
    session_types_encoded = torch.tensor([type_encoder.transform(session_types)], dtype=torch.long)
    session_encoded = torch.cat((session_encoded.unsqueeze(-1), session_types_encoded.unsqueeze(-1)), -1)
    with torch.no_grad():
        output = model(session_encoded)
        predicted = torch.argmax(output).item()
        recommended_item = item_encoder.inverse_transform([predicted])[0]
        return recommended_item

# Example: Get a user input and recommend the next item
user_session = []
user_session_types = []
print("Enter 5 item ids and item types, one at a time:")
for i in range(5):
    item_id = int(input(f"Item {i + 1}: "))
    item_type = input(f"Item type {i + 1}: ")
    user_session.append(item_id)
    user_session_types.append(item_type)

recommended_item = recommend_next_item(user_session, user_session_types)
print(f"Recommended next item: {recommended_item}")
