import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R

# Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# List of paths to your datasets
data_paths = [
    './ground_truth_estimate/data_aug_mh01.csv',
    './ground_truth_estimate/data_aug_mh02.csv',
    './ground_truth_estimate/data_aug_mh03.csv'
    #'./ground_truth_estimate/data_aug_mh04.csv',
    #'./ground_truth_estimate/data_aug_mh05.csv'
]

# Define sequence length
sequence_length = 15

# Initialize lists to store sequences, biases, and positions from all datasets
all_sequences = []
all_biases = []
all_positions = []
all_velocities = []

# Normalize input features and bias values
def normalize(data):
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True)
    return (data - mean) / std, mean, std

def normalize_bias(bias):
    mean = bias.mean(axis=0)
    std = bias.std(axis=0)
    return (bias - mean) / std, mean, std

def denormalize(data, mean, std):
    return data * std + mean

def denormalize_bias(bias, mean, std):
    return bias * std + mean


print('----Organizing Data----')

# Process each dataset separately
for data_path in data_paths:
    data = pd.read_csv(data_path)
    
    sequences = []
    positions = []
    velocities = []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length][[
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
            'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
            ' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]'
        ]].values
        sequences.append(sequence)
        position = data.iloc[i + sequence_length][[
            ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]'
        ]].values
        velocity = data.iloc[i + sequence_length][[
            ' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]'
        ]].values
        positions.append(position)
        velocities.append(velocity)
    
    sequences = np.array(sequences)
    positions = np.array(positions)
    velocities = np.array(velocities)
    X = sequences[:, :, :-3]  # Acceleration, angular velocity, previous gyro bias
    y = sequences[:, -1, -3:]  # Acceleration bias (b^a)
    
    # Normalize input features and bias values for this dataset
    X, X_mean, X_std = normalize(X)
    y, bias_mean, bias_std = normalize_bias(y)
    
    # Append to the global lists
    all_sequences.append(X)
    all_biases.append(y)
    all_positions.append(positions)
    all_velocities.append(velocities)  # Store velocities

# Concatenate all sequences, biases, positions, and velocities
X_all = np.concatenate(all_sequences, axis=0)
y_all = np.concatenate(all_biases, axis=0)
positions_all = np.concatenate(all_positions, axis=0)
velocities_all = np.concatenate(all_velocities, axis=0)  # Concatenate velocities

print('\n----Organizing Complete!----\nsize: ')
print('X_all:', X_all.shape)
print('y_all:', y_all.shape)
print('positions_all:', positions_all.shape)
print('velocities_all:', velocities_all.shape)

# Convert to PyTorch tensors
X_all = torch.tensor(X_all, dtype=torch.float32)
y_all = torch.tensor(y_all, dtype=torch.float32)

# Total number of sequences
total_sequences = len(X_all)

# Define the number of sequences for testing
num_test_sequences = 30

# Define the indices for splitting the data
indices = np.arange(total_sequences)
np.random.shuffle(indices)

# Split indices
test_indices = indices[:num_test_sequences]
remaining_indices = indices[num_test_sequences:]

# Split remaining indices into training and validation sets
train_size = int(len(remaining_indices) * 0.9)
train_indices = remaining_indices[:train_size]
val_indices = remaining_indices[train_size:]

# Create training, validation, and testing sets
X_train = X_all[train_indices]
y_train = y_all[train_indices]
positions_train = positions_all[train_indices]
velocities_train = velocities_all[train_indices]  # Training velocities

X_val = X_all[val_indices]
y_val = y_all[val_indices]
positions_val = positions_all[val_indices]
velocities_val = velocities_all[val_indices]  # Validation velocities

X_test = X_all[test_indices]
y_test = y_all[test_indices]
positions_test = positions_all[test_indices]
velocities_test = velocities_all[test_indices]  # Test velocities


# Step 2: Define the Transformer model using TransformerEncoder
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over the sequence length
        x = self.fc(x)
        return x
    

##############################################################################################
#
#                                       HYPERPARAMETER                                       #                               
#
##############################################################################################
# Model hyperparameters
input_dim = 6   # 3 (acc) + 3 (gyro)
model_dim = 64  # Model dimension
num_heads = 8   # Number of heads (# of self-attention)
num_layers = 6  # Number of encoder layers
output_dim = 3  # Acceleration bias (3D)
num_epochs = 30 # Number of epochs
lr = 0.1      # Learning rate
# Scheduler hyperparamter
step_size = 10
gamma = 0.1

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# Train
class AccelBiasDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloader for training
train_dataset = AccelBiasDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create dataset and dataloader for validation
val_dataset = AccelBiasDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Function to compute position using the given equation
def compute_position_with_correction(current_position, velocity, acceleration, bias, R_t, g, dt=0.01, disturbance=0.0):
    acceleration_corrected = acceleration - bias - disturbance
    new_position = (current_position + velocity * dt + 0.5 * g * dt ** 2 +
                    0.5 * R_t @ (acceleration_corrected) * dt ** 2)
    return new_position

# Normalize disturbance (if needed, set to zero for simplicity)
disturbance = np.zeros(3)

# Function to denormalize acceleration
def denormalize_acceleration(acceleration, mean, std):
    mean = mean[0, 0, :3]  # Extract mean for acceleration
    std = std[0, 0, :3]    # Extract std for acceleration
    return acceleration * std + mean

# Function to compute velocity using the given equation
def compute_velocity_with_correction(current_velocity, acceleration, bias, R_t, g, dt=0.01, disturbance=0.0):
    acceleration_corrected = acceleration - bias - disturbance
    new_velocity = (current_velocity + g * dt +
                    R_t @ (acceleration_corrected) * dt)
    return new_velocity

alpha = 1000
print(f'\n#############################\n       Alpha = {alpha}\n#############################\n')

print('----Start Training----')

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_index, (batch_X, batch_y) in enumerate(train_dataloader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Transfer data to GPU
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss_bias = criterion(outputs, batch_y)
        
        # Compute position loss
        batch_size = batch_X.size(0)
        position_losses = []
        
        for i in range(batch_size):
            acceleration_normalized = batch_X[i, -1, :3].cpu().numpy()  # Last acceleration in sequence (normalized)
            acceleration = denormalize_acceleration(acceleration_normalized, X_mean, X_std)
            
            # Calculate the actual index in the original dataset
            actual_index = train_indices[batch_index * train_dataloader.batch_size + i]

            true_position = positions_all[actual_index]
            predicted_bias = outputs[i].cpu().detach().numpy()

            # Get the initial position and velocity for the sequence
            initial_index = max(actual_index - (sequence_length - 1), 0)
            initial_position = positions_all[initial_index]
            initial_velocity = velocities_all[initial_index]  # Use stored velocities
            R_t = np.eye(3)  # Assume identity matrix for rotation for simplicity
            g = np.array([0, 0, -9.81])  # Gravity vector

            # Compute velocity
            computed_velocity = compute_velocity_with_correction(
                initial_velocity, acceleration, predicted_bias, R_t, g, dt=0.01, disturbance=disturbance
            )

            # Compute position using the computed velocity
            computed_position = compute_position_with_correction(
                initial_position, computed_velocity, acceleration, predicted_bias, R_t, g, dt=0.01, disturbance=disturbance
            )

            position_loss = mean_squared_error(true_position, computed_position)
            position_losses.append(position_loss)
        
        loss_position = torch.tensor(position_losses, dtype=torch.float32, device=device).mean()
        
        # Combine losses
        loss = loss_bias + alpha * loss_position
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    scheduler.step()  # Step the scheduler at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # Optionally, evaluate on the validation set at the end of each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss_bias = criterion(outputs, batch_y)
            val_loss += loss_bias.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
print('----Finished Training----')
    
# After training, evaluate on the test set
test_dataset = AccelBiasDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss_bias = criterion(outputs, batch_y)
        test_loss += loss_bias.item()
avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss: {avg_test_loss:.4f}')


# Denormalize predicted biases
predicted_biases = denormalize_bias(outputs.cpu().numpy(), bias_mean, bias_std)

# Compare the computed position to the ground truth position for test set
initial_position = positions_test[0]  # Starting position from the test set

total_bias_mse = 0
total_position_error = 0

for i in range(len(X_test)):
    acceleration_normalized = X_test[i, -1, :3].cpu().numpy()  # Last acceleration in sequence (normalized)
    acceleration = denormalize_acceleration(acceleration_normalized, X_mean, X_std)
    true_position = positions_test[i]
    predicted_bias = predicted_biases[i]
    current_position = initial_position if i == 0 else computed_position
    current_velocity = velocities_test[i]  # Use stored velocities
    R_t = np.eye(3)  # Assume identity matrix for rotation for simplicity
    g = np.array([0, 0, -9.81])  # Gravity vector

    # Compute velocity
    computed_velocity = compute_velocity_with_correction(
        current_velocity, acceleration, predicted_bias, R_t, g, dt=0.01, disturbance=disturbance
    )

    # Compute position using the computed velocity
    computed_position = compute_position_with_correction(
        current_position, computed_velocity, acceleration, predicted_bias, R_t, g, dt=0.01, disturbance=disturbance
    )
    
    bias_mse = mean_squared_error(y_test[i].cpu().numpy(), predicted_bias)
    position_error = mean_squared_error(true_position, computed_position)
    
    total_bias_mse += bias_mse
    total_position_error += position_error
    
    print(f'Sequence {i+1}:')
    print(f'  Predicted Bias: {predicted_bias}')
    print(f'  True Bias: {y_test[i].cpu().numpy()}')
    print(f'  True Position: {true_position}')
    print(f'  Computed Position: {computed_position}')
    print(f'  MSE Bias: {bias_mse}')
    print(f'  Position Error: {position_error}')

    # Update initial_position and initial_velocity for the next iteration
    initial_position = computed_position
    initial_velocity = computed_velocity
    
# Calculate average errors
net_average_bias_mse = total_bias_mse / len(X_test)
net_average_position_error = total_position_error / len(X_test)

print(f'Net Average Bias MSE: {net_average_bias_mse}')
print(f'Net Average Position Error: {net_average_position_error}')