import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
data_path = 'ground_truth_estimate\data_aug.csv'
data = pd.read_csv(data_path)

#print(data.head())
#print(data.columns)

sequences = []
sequence_length = 10  # Adjusted sequence length based on typical time-series data

for i in range(len(data) - sequence_length):
    sequence = data.iloc[i:i+sequence_length][[
        'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]', 
        'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 
        ' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]'
    ]].values
    sequences.append(sequence)

sequences = np.array(sequences)
X = sequences[:, :, :-3]  # acceleration, angular velocity, previous gyro bias
y = sequences[:, -1, -3:]  # gyro bias

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

#print(X)
#print(y)

# Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Train split
X_train = X[100:]
y_train = y[100:]

print(y_train)

# Step 2: Define the Transformer model using TransformerEncoder
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over the sequence length
        x = self.fc(x)
        return x

# Model hyperparameters
input_dim = 6   # 3 (acc) + 3 (gyro)
model_dim = 64  # Model dimension
num_heads = 8   # Number of heads (# of self-attention)
num_layers = 6  # Number of encoder-decoder layers (cross-attention)
output_dim = 3  # Gyro bias (3D)
num_epochs = 50 # Number of epoch
lr = 0.0005     # Learning rate

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# Train
class GyroBiasDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = GyroBiasDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Custom Loss Function
class CustomLossWithCovariance(nn.Module):
    def __init__(self, sigma_init=2.5e-3, epsilon=1e-6, device='cpu'):
        super(CustomLossWithCovariance, self).__init__()
        self.sigma = torch.eye(3) * sigma_init  # Initialize covariance matrix
        self.sigma += epsilon * torch.eye(3)  # Add epsilon to ensure positive definiteness
        self.sigma = self.sigma.to(device)  # Move sigma to the specified device

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        sigma_inv = torch.inverse(self.sigma)  # Inverse of the covariance matrix
        log_det_sigma = torch.logdet(self.sigma)  # Log determinant of the covariance matrix

        mahalanobis_distances = []
        for i in range(batch_size):
            diff = predictions[i] - targets[i]
            diff = diff.to(sigma_inv.device)  # Ensure diff is on the same device as sigma_inv
            mahalanobis_distance = torch.matmul(diff.unsqueeze(0), sigma_inv)
            mahalanobis_distance = torch.matmul(mahalanobis_distance, diff.unsqueeze(1)).squeeze()
            mahalanobis_distances.append(mahalanobis_distance)

        mahalanobis_distances = torch.stack(mahalanobis_distances)
        loss = abs(log_det_sigma + mahalanobis_distances.mean())

        return loss
    
    
#criterion = nn.MSELoss()
criterion = CustomLossWithCovariance(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Transfer data to GPU
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Infer
def infer_gyro_bias(model, sequences):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequence in sequences:
            sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_bias = model(sequence).squeeze(0).cpu().numpy()
            predictions.append(predicted_bias)
    return np.array(predictions)

# Get the first 100 sequences
first_100_sequences = X[:100].numpy()

# Predict gyro bias for the first 100 sequences
predicted_biases = infer_gyro_bias(model, first_100_sequences)

# True gyro bias values for the first 100 sequences
true_biases = y[:100].numpy()

# Calculate Mean Squared Error between predicted and true values
mse = mean_squared_error(true_biases, predicted_biases)
print(f'Mean Squared Error for the first 100 sequences: {mse}')

# Print the predicted biases and true biases for comparison
for i, (predicted_bias, true_bias) in enumerate(zip(predicted_biases, true_biases)):
    print(f'Sequence {i+1}:')
    print(f'  Pred: {predicted_bias}')
    print(f'  True: {true_bias}')
    print(f'  MSE: {mean_squared_error(true_bias, predicted_bias)}')
    
    
# Step 5: Infer the gyro bias values for the last 10 sequences
last_10_sequences = X[-10:].numpy()

# Predict gyro bias for the last 10 sequences
predicted_biases_last_10 = infer_gyro_bias(model, last_10_sequences)

# True gyro bias values for the last 10 sequences
true_biases_last_10 = y[-10:].numpy()

# Calculate Mean Squared Error between predicted and true values for the last 10 sequences
mse_last_10 = mean_squared_error(true_biases_last_10, predicted_biases_last_10)
print(f'Mean Squared Error for the last 10 sequences: {mse_last_10}')

# Print the predicted biases and true biases for the last 10 sequences
for i, (predicted_bias, true_bias) in enumerate(zip(predicted_biases_last_10, true_biases_last_10)):
    print(f'Sequence {len(X)-10+i+1}:')
    print(f'  Pred: {predicted_bias}')
    print(f'  True: {true_bias}')
    print(f'  MSE: {mean_squared_error(true_bias, predicted_bias)}')