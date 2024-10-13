import pandas as pd
import numpy as np
import torch
import math
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import KFold
from tqdm import tqdm

#### Setting Dataset Type ####

dataset = 'euroc'
#dataset = 'kitti'

# Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# List of paths to your datasets
if dataset == 'euroc':
    dt = 0.005
    train_val_data_paths = [
        './EuRoC Dataset/data_aug_mh01.csv',
        './EuRoC Dataset/data_aug_mh03.csv',
        './EuRoC Dataset/data_aug_mh05.csv',
        
        './EuRoC Dataset/data_aug_v102.csv',
        './EuRoC Dataset/data_aug_v201.csv',
        './EuRoC Dataset/data_aug_v203.csv'
    ]
elif dataset == 'kitti':
    dt = 0.1
    train_val_data_paths = [
        './kitti/augmented_data/kitti_aug_00.csv',
        './kitti/augmented_data/kitti_aug_01.csv',
        './kitti/augmented_data/kitti_aug_02.csv',
        #'./kitti/augmented_data/kitti_aug_03.csv',
        './kitti/augmented_data/kitti_aug_04.csv',
        './kitti/augmented_data/kitti_aug_05.csv',
        './kitti/augmented_data/kitti_aug_06.csv',
        #'./kitti/augmented_data/kitti_aug_07.csv', test
        './kitti/augmented_data/kitti_aug_08.csv',
        './kitti/augmented_data/kitti_aug_09.csv'
        #'./kitti/augmented_data/kitti_aug_10.csv'  test
    ]
else:
    raise Exception('Wrong data type!!')


# Define sequence length
sequence_length = 10

# Initialize lists to store sequences, biases, and quaternions from all datasets
all_sequences = []
all_y = []
all_biases = []
all_quaternions = []
all_positions = []
all_velocities = []
all_true_biases = []
all_next_positions = []

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

# Function to process dataset
def process_dataset(data_path):
    data = pd.read_csv(data_path)
    
    sequences = []
    biases = []
    true_biases = []
    quaternions = []
    positions = []
    next_positions = []
    velocities = []
    for i in range(len(data) - sequence_length - 2):
        sequence = data.iloc[i+1:i+sequence_length+1][[
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
            'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]'
        ]].values
        sequences.append(sequence)
        
        bias = data.iloc[i:i+sequence_length][[
            ' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]'
            #'true_bias_accel_x', 'true_bias_accel_y', 'true_bias_accel_z'
        ]].values
        biases.append(bias)
        
        true_bias = data.iloc[i+1:i+sequence_length+1][[
            #'true_bias_accel_x', 'true_bias_accel_y', 'true_bias_accel_z'
            ' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]'
        ]].values
        true_biases.append(true_bias)
        
        quaternion = data.iloc[i + sequence_length+1][[
            ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'
        ]].values
        quaternions.append(quaternion)

        position = data.iloc[i + sequence_length+1][[
            ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]'
        ]].values
        positions.append(position)

        next_position = data.iloc[i + sequence_length + 2][[
            ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]'
        ]].values
        next_positions.append(next_position)

        velocity = data.iloc[i + sequence_length + 1][[
            ' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]'
        ]].values
        velocities.append(velocity)
    
    sequences = np.array(sequences)
    biases = np.array(biases)
    true_biases = np.array(true_biases)
    quaternions = np.array(quaternions)
    positions = np.array(positions)
    velocities = np.array(velocities)
    next_positions = np.array(next_positions)
    X = sequences  # Acceleration, angular velocity
    y = true_biases[:, -1, :]  # Last bias in sequence
    
    return X, y, quaternions, biases, positions, velocities, next_positions

# Process each dataset separately
for data_path in train_val_data_paths:
    # Extract dataset name from file path for saving/loading and append "accel"
    dataset_name = os.path.basename(data_path).split('.')[0]  # e.g., 'data_aug_mh01'
    processed_data_path = f'./processed_{dataset_name}_accel.pth'

    if os.path.exists(processed_data_path):
        print(f'----Loading Preprocessed Data for {dataset_name} (accel)----')
        # Load preprocessed data
        checkpoint = torch.load(processed_data_path, map_location=device)

        X = checkpoint['X']
        y = checkpoint['y']
        quaternions = checkpoint['quaternions']
        biases = checkpoint['biases']
        positions = checkpoint['positions']
        velocities = checkpoint['velocities']
        next_positions = checkpoint['next_positions']

    else:
        print(f'----Processing Data for {dataset_name} (accel)----')
        X, y, quaternions, biases, positions, velocities, next_positions = process_dataset(data_path)

        # Save the processed data
        print(f'----Saving Preprocessed Data for {dataset_name} (accel)----')
        torch.save({
            'X': X,
            'y': y,
            'quaternions': quaternions,
            'biases': biases,
            'positions': positions,
            'velocities': velocities,
            'next_positions': next_positions,
        }, processed_data_path)

    # Append to the global lists
    all_sequences.append(X)
    all_y.append(y)
    all_quaternions.append(quaternions)
    all_biases.append(biases)
    all_positions.append(positions)
    all_velocities.append(velocities)
    all_next_positions.append(next_positions)

# Concatenate all sequences, biases, and quaternions
X_all = np.concatenate(all_sequences, axis=0)
y_all = np.concatenate(all_y, axis=0)
quaternions_all = np.concatenate(all_quaternions, axis=0)
biases_all = np.concatenate(all_biases, axis=0)
positions_all = np.concatenate(all_positions, axis=0)
velocities_all = np.concatenate(all_velocities, axis=0)
next_positions_all = np.concatenate(all_next_positions, axis=0)


# Reshape biases to apply RobustScaler (into 2D array: (n_samples * sequence_length, n_features))
n_samples, sequence_len, n_features = biases_all.shape
biases_reshaped = biases_all.reshape(-1, n_features)

# Apply RobustScaler globally over the entire dataset
scaler = RobustScaler()
biases_scaled = scaler.fit_transform(biases_reshaped)

# Reshape scaled biases back to original 3D shape (n_samples, sequence_len, n_features)
biases_scaled = biases_scaled.reshape(n_samples, sequence_len, n_features)

# Now, biases_all is the globally scaled version
biases_all = biases_scaled

# Calculate and print median and IQR for original biases (before scaling)
median = np.median(biases_reshaped, axis=0)
Q1 = np.percentile(biases_reshaped, 25, axis=0)
Q3 = np.percentile(biases_reshaped, 75, axis=0)
IQR = Q3 - Q1

print("Median of biases (before scaling):", median)
print("IQR of biases (before scaling):", IQR)

# Convert median and IQR to PyTorch tensors
bias_median = torch.tensor(median, dtype=torch.float32, device=device)
bias_iqr = torch.tensor(IQR, dtype=torch.float32, device=device)

X_mean = X_all.mean(axis=(0, 1))
X_std = X_all.std(axis=(0, 1))
#X_all = (X_all - X_mean) / X_std

bias_mean = biases_all.mean(axis=(0, 1))
bias_std = biases_all.std(axis=(0, 1))
#biases_all = (biases_all - bias_mean) / bias_std

#pd.DataFrame(X_all.reshape(X_all.shape[0], -1)).to_csv('X_all.csv', index=False)
#pd.DataFrame(y_all).to_csv('y_all.csv', index=False)
#pd.DataFrame(quaternions_all).to_csv('quaternions_all.csv', index=False)
#pd.DataFrame(biases_all.reshape(biases_all.shape[0], -1)).to_csv('biases_all.csv', index=False)
#pd.DataFrame(positions_all).to_csv('positions_all.csv', index=False)
#pd.DataFrame(velocities_all).to_csv('velocities_all.csv', index=False)
#pd.DataFrame(next_positions_all).to_csv('next_positions_all.csv', index=False)


# Convert to PyTorch tensors
X_all = torch.tensor(X_all, dtype=torch.float32, device=device)
y_all = torch.tensor(y_all, dtype=torch.float32, device=device)
biases_all = torch.tensor(biases_all, dtype=torch.float32, device=device)

X_mean = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_std = torch.tensor(X_std, dtype=torch.float32, device=device)
bias_mean = torch.tensor(bias_mean, dtype=torch.float32, device=device)
bias_std = torch.tensor(bias_std, dtype=torch.float32, device=device)

quaternions_all = torch.tensor(quaternions_all, dtype=torch.float32, device=device)
biases_all = torch.tensor(biases_all, dtype=torch.float32, device=device)
positions_all = torch.tensor(positions_all, dtype=torch.float32, device=device)
velocities_all = torch.tensor(velocities_all, dtype=torch.float32, device=device)
next_positions_all = torch.tensor(next_positions_all, dtype=torch.float32, device=device)


print('\n----Organizing Complete!----\nsize: ')
print('X_all:', X_all.shape)
print('y_all:', y_all.shape)
print('biases_all:', biases_all.shape)
print('quaternions_all:', quaternions_all.shape)
print('velocities_all:', velocities_all.shape)
print('positions_all:', positions_all.shape)
print('X_mean:', X_mean)
print('X_std:', X_std)
print('Bias_mean:', bias_mean)
print('Bias_std:', bias_std)


# Define the positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Define the Transformer model using TransformerEncoder and TransformerDecoder
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.decoder_embedding = nn.Linear(output_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        #src = self.positional_encoding(src)
        tgt = self.decoder_embedding(tgt)
        #tgt = self.positional_encoding(tgt)
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc(output)
        return output[:, -1, :]  # Return only the last element of the sequence

# Model hyperparameters
input_dim = 6   # 3 (acc) + 3 (gyro)
model_dim = 256  # Model dimension
num_heads = 8   # Number of heads (# of self-attention)
num_layers = 2  # Number of encoder-decoder layers (cross-attention)
output_dim = 3  # Gyro bias (3D)
num_epochs = 10 # Number of epochs
lr = 1e-5      # Learning rate
batch_size = 32
# Scheduler hyperparamter
step_size = 5
gamma = 0.1

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# Function to initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Initialize using Xavier/Glorot uniform initialization
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Set bias to zero
    elif isinstance(m, nn.Conv2d):
        # Initialize Conv layers similarly if you have any
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.TransformerEncoderLayer) or isinstance(m, nn.TransformerDecoderLayer):
        # Transformer layers initialization (often PyTorch default is sufficient)
        for param in m.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

# Apply weight initialization to the model
model.apply(initialize_weights)

# Train
class GyroBiasDataset(Dataset):
    def __init__(self, X, y, biases):
        self.X = X
        self.y = y
        self.biases = biases

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.biases[idx]

def compute_acceleration_with_correction(current_position, acceleration, bias, velocity, gravity, quaternion, dt=0.005, disturbance=0.0):
    acceleration_corrected = acceleration - bias - disturbance
    rotated_term = 0.5 * quaternion_rotate(quaternion, acceleration_corrected) * (dt**2)

    return current_position + velocity * dt + 0.5 * gravity * (dt ** 2) + rotated_term

def quaternion_product(q, r):
    # Compute quaternion product (ensure it preserves gradients)
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return torch.stack([
        q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
        q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
        q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
        q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0
    ], dim=0)

def quaternion_rotate(q, v):
    # Handle the case where q is 1D (for a single quaternion) and v is 1D (for a single vector)
    if q.dim() == 1:
        q_conj = torch.cat([q[:1], -q[1:]], dim=0)  # Create quaternion conjugate for 1D tensor
        v_quat = torch.cat([torch.zeros(1, device=v.device), v], dim=0)  # Convert vector to quaternion form
        rotated_v = quaternion_product(quaternion_product(q, v_quat), q_conj)
        return rotated_v[1:]  # Return only the vector part
    # Handle the case where q is 2D (batch of quaternions) and v is 2D (batch of vectors)
    elif q.dim() == 2:
        q_conj = torch.cat([q[:, :1], -q[:, 1:]], dim=1)  # Create quaternion conjugate for 2D tensor
        v_quat = torch.cat([torch.zeros((v.shape[0], 1), device=v.device), v], dim=1)
        rotated_v = quaternion_product(quaternion_product(q, v_quat), q_conj)
        return rotated_v[:, 1:]  # Return the rotated vector part
    else:
        raise ValueError(f"Unexpected dimension for q: {q.dim()}")

# Normalize disturbance (if needed, set to zero for simplicity)
disturbance = torch.zeros(3, device=device)

# Function to denormalize angular velocity
def denormalize_angular_velocity(angular_velocity, mean, std):
    mean = mean[3:6]
    std = std[3:6]
    return angular_velocity * std + mean

def denormalize_acceleration(acceleration, mean, std):
    mean = mean[0:3]
    std = std[0:3]
    return acceleration * std + mean

class CustomPositionLoss(nn.Module):
    def __init__(self):
        super(CustomPositionLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(delta=1.0)

    def forward(self, true_quaternions, true_positions, predicted_biases, batch_X, positions_all, velocities_all, gravity_vector, indices, sequence_length, dt=0.005, disturbance=0.0):
        batch_size = true_positions.size(0)
        position_losses = []

        for i in range(batch_size):
            #acceleration_normalized = batch_X[i, -1, 0:3]
            #acceleration = denormalize_acceleration(acceleration_normalized, X_mean, X_std)
            acceleration = batch_X[i, -1, 0:3]
            
            
            true_quaternion = true_quaternions[i]
            true_position = true_positions[i]

            # Calculate the actual index in the original dataset
            actual_index = indices[i]

            # Get the initial quaternion for the sequence
            initial_index = max(actual_index - (sequence_length - 1), 0)
            
            # Removed .detach() to allow gradients to flow through
            #initial_position = positions_all[initial_index].clone().to(device).requires_grad_(True)
            #initial_velocity = velocities_all[initial_index].clone().to(device).requires_grad_(True)
            initial_position = positions_all[initial_index].to(device)
            initial_velocity = velocities_all[initial_index].to(device)
            

            predicted_position = compute_acceleration_with_correction(
                initial_position, 
                acceleration, 
                predicted_biases[i], 
                initial_velocity, 
                gravity_vector.to(device), 
                true_quaternion, 
                dt=dt, 
                disturbance=disturbance
            )

            position_loss = self.huber_loss(predicted_position, true_position)
            
            
            position_losses.append(position_loss)

        if position_losses:  # Check if quaternion_losses is not empty
            position_losses = torch.stack(position_losses)
            loss_position = position_losses.mean()
        else:
            print("Warning: position_losses is empty.")
            loss_position = torch.tensor(0.0, dtype=torch.float32, device=batch_X.device, requires_grad=True)

        return loss_position
    
#criterion = CustomPositionLoss()
criterion = nn.MSELoss()
#criterion = nn.HuberLoss(delta=1.0)

# K-Fold Cross Validation
k_folds = 20
kf = KFold(n_splits=k_folds, shuffle=True)

print('----Start Training (Accel)----')

fold_results = {}

'''
for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f'Fold {fold+1}/{k_folds}')

    # Split data
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    biases_train = biases_all[train_idx]
    biases_val = biases_all[val_idx]
    quaternions_train = quaternions_all[train_idx]
    quaternions_val = quaternions_all[val_idx]
    positions_train = positions_all[train_idx]
    positions_val = positions_all[val_idx]
    true_positions_train = next_positions_all[train_idx]
    true_positions_val = next_positions_all[val_idx]

    # Gravity
    gravity_vector = torch.tensor([0, 0, -9.81], device=device)

    train_dataset = GyroBiasDataset(X_train, y_train, biases_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = GyroBiasDataset(X_val, y_val, biases_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_index, (batch_X, batch_y, batch_biases) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=True)):
            batch_X, batch_y, batch_biases = batch_X.to(device), batch_y.to(device), batch_biases.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, batch_biases)
            
            start_idx = batch_index * train_dataloader.batch_size
            end_idx = start_idx + batch_X.size(0)
            true_quaternions = quaternions_train[start_idx:end_idx].to(device)
            true_positions = positions_train[start_idx:end_idx].to(device)
            loss = criterion(true_quaternions, true_positions, outputs, batch_X, positions_all, velocities_all, gravity_vector, train_idx[start_idx:end_idx], sequence_length)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.14f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_index, (batch_X, batch_y, batch_biases) in enumerate(tqdm(val_dataloader, desc="Validating", leave=True)):
                batch_X, batch_y, batch_biases = batch_X.to(device), batch_y.to(device), batch_biases.to(device)
                outputs = model(batch_X, batch_biases)

                start_idx = batch_index * val_dataloader.batch_size
                end_idx = start_idx + batch_X.size(0)
                true_quaternions = quaternions_val[start_idx:end_idx].to(device)
                true_positions = positions_val[start_idx:end_idx].to(device)
                loss_quaternion = criterion(true_quaternions, true_positions, outputs, batch_X, positions_all, velocities_all, gravity_vector, val_idx[start_idx:end_idx], sequence_length)

                val_loss += loss_quaternion.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation Loss: {avg_val_loss:.20f}')

    fold_results[fold] = avg_val_loss
'''

def denormalize_model_output_iqr(scaled_output, bias_median, bias_iqr):
    return scaled_output * bias_iqr + bias_median

# Non K-Fold version

# Get the first train-test split
train_idx, val_idx = next(kf.split(X_all))

# Split data
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
biases_train = biases_all[train_idx]
biases_val = biases_all[val_idx]
quaternions_train = quaternions_all[train_idx]
quaternions_val = quaternions_all[val_idx]
positions_train = positions_all[train_idx]
positions_val = positions_all[val_idx]
velocities_train = velocities_all[train_idx] 
velocities_val = velocities_all[val_idx]
true_positions_train = next_positions_all[train_idx]
true_positions_val = next_positions_all[val_idx]

# Gravity
gravity_vector = torch.tensor([0, 0, -9.81], device=device)

train_dataset = GyroBiasDataset(X_train, y_train, biases_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GyroBiasDataset(X_val, y_val, biases_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

data_counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_index, (batch_X, batch_y, batch_biases) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=True)):
        batch_X, batch_y, batch_biases = batch_X.to(device), batch_y.to(device), batch_biases.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, batch_biases)

        # Compute MSE loss between predicted and true biases
        loss = criterion(outputs, batch_y)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

            

    avg_loss = epoch_loss / len(train_dataloader)
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.14f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_index, (batch_X, batch_y, batch_biases) in enumerate(tqdm(val_dataloader, desc="Validating", leave=True)):
            batch_X, batch_y, batch_biases = batch_X.to(device), batch_y.to(device), batch_biases.to(device)
            outputs = model(batch_X, batch_biases)

            # Compute MSE loss for validation
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss:.20f}')

fold_results[0] = avg_val_loss



print('----Finished Training----')

# Save the trained model
model.to('cpu')

scripted_model = torch.jit.script(model)
if dataset == 'euroc':
    model_path = "euroc_scripted_accel_model_10_mse.pth"
elif dataset == 'kitti':
    model_path = "kitti_scripted_accel_model_10_mse.pth"
#torch.save(model.state_dict(), model_path)
scripted_model.save(model_path)

print(f"Model saved to {model_path}")

# Print fold results
for fold, loss in fold_results.items():
    print(f'Fold {fold+1}: Validation Loss = {loss}')

