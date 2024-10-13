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

##############################


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
all_true_biases = []
all_next_quaternions = []

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
    raw_data = pd.read_csv(data_path)
    
    data = raw_data[~raw_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
                       'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
                       #' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]',
                       'true_bias_gyro_x', 'true_bias_gyro_y', 'true_bias_gyro_z',
                       ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'
                        ]].isnull().any(axis=1)]
    
    sequences = []
    biases = []
    true_biases = []
    quaternions = []
    next_quaternions = []
    for i in range(len(data) - sequence_length - 2):
        sequence = data.iloc[i+1:i+sequence_length+1][[
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
            'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]'
        ]].values
        sequences.append(sequence)
        
        bias = data.iloc[i:i+sequence_length][[
            #' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]'
            'true_bias_gyro_x', 'true_bias_gyro_y', 'true_bias_gyro_z'
        ]].values
        biases.append(bias)
        
        true_bias = data.iloc[i+1:i+sequence_length+1][[
            #' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]'
            'true_bias_gyro_x', 'true_bias_gyro_y', 'true_bias_gyro_z'
        ]].values
        true_biases.append(true_bias)
        
        quaternion = data.iloc[i + sequence_length][[
            ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'
        ]].values
        quaternions.append(quaternion)

        next_quaternion = data.iloc[i + sequence_length + 2][[
            ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'
        ]].values
        next_quaternions.append(next_quaternion)


    
    sequences = np.array(sequences)
    biases = np.array(biases)
    true_biases = np.array(true_biases)
    quaternions = np.array(quaternions)
    next_quaternions = np.array(next_quaternions)
    
    X = sequences  # Acceleration, angular velocity
    y = true_biases[:, -1, :]  # Last bias in sequence
    
    return X, y, quaternions, biases, next_quaternions

# Process each dataset separately
for data_path in train_val_data_paths:
    # Extract dataset name from file path for saving/loading and append "gyro"
    dataset_name = os.path.basename(data_path).split('.')[0]  # e.g., 'data_aug_mh01'
    processed_data_path = f'./processed_{dataset_name}_gyro.pth'

    if os.path.exists(processed_data_path):
        print(f'----Loading Preprocessed Data for {dataset_name} (gyro)----')
        # Load preprocessed data
        checkpoint = torch.load(processed_data_path, map_location=device)

        X = checkpoint['X']
        y = checkpoint['y']
        quaternions = checkpoint['quaternions']
        biases = checkpoint['biases']
        next_quaternions = checkpoint['next_quaternions']

    else:
        print(f'----Processing Data for {dataset_name} (gyro)----')
        X, y, quaternions, biases, next_quaternions = process_dataset(data_path)

        # Save the processed data
        print(f'----Saving Preprocessed Data for {dataset_name} (gyro)----')
        torch.save({
            'X': X,
            'y': y,
            'quaternions': quaternions,
            'biases': biases,
            'next_quaternions': next_quaternions,
        }, processed_data_path)

    # Append to the global lists
    all_sequences.append(X)
    all_y.append(y)
    all_quaternions.append(quaternions)
    all_biases.append(biases)
    all_next_quaternions.append(next_quaternions)


# Concatenate all sequences, biases, and quaternions
X_all = np.concatenate(all_sequences, axis=0)
y_all = np.concatenate(all_y, axis=0)
quaternions_all = np.concatenate(all_quaternions, axis=0)
biases_all = np.concatenate(all_biases, axis=0)
next_quaternions_all = np.concatenate(all_next_quaternions, axis=0)

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


# Convert to PyTorch tensors
X_all = torch.tensor(X_all, dtype=torch.float32, device=device)
y_all = torch.tensor(y_all, dtype=torch.float32, device=device)
biases_all = torch.tensor(biases_all, dtype=torch.float32, device=device)

#X_mean = X_all.mean(axis=(0, 1))
#X_std = X_all.std(axis=(0, 1))
#X_all = (X_all - X_mean) / X_std

bias_mean = biases_all.mean(axis=(0, 1))
bias_std = biases_all.std(axis=(0, 1))
#biases_all = (biases_all - bias_mean) / bias_std

#X_mean = torch.tensor(X_mean, dtype=torch.float32, device=device)
#X_std = torch.tensor(X_std, dtype=torch.float32, device=device)
bias_mean = torch.tensor(bias_mean, dtype=torch.float32, device=device)
bias_std = torch.tensor(bias_std, dtype=torch.float32, device=device)

quaternions_all = torch.tensor(quaternions_all, dtype=torch.float32, device=device)
next_quaternions_all = torch.tensor(next_quaternions_all, dtype=torch.float32, device=device)

print('\n----Organizing Complete!----\nsize: ')
print('X_all:', X_all.shape)
print('y_all:', y_all.shape)
print('biases_all:', biases_all.shape)
print('quaternions_all:', quaternions_all.shape)
#print('X_mean:', ['{:.8f}'.format(x) for x in X_mean])
#print('X_std:', ['{:.8f}'.format(x) for x in X_std])
print('Bias_mean:', ['{:.8f}'.format(x) for x in bias_mean])
print('Bias_std:', ['{:.8f}'.format(x) for x in bias_std])




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
        src = self.positional_encoding(src)
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc(output)
        return output[:, -1, :]  # Return only the last element of the sequence



##############################################################################################
#z
#                                       HYPERPARAMETER                                       #                               
#
##############################################################################################
# Model hyperparameters
input_dim = 6   # 3 (acc) + 3 (gyro)
model_dim = 256  # Model dimension
num_heads = 8   # Number of heads (# of self-attention)
num_layers = 2  # Number of encoder-decoder layers (cross-attention)
output_dim = 3  # Gyro bias (3D)
num_epochs = 10 # Number of epochs
lr = 1e-8      # Learning rate
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

#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Function to compute quaternion using the given equation
def compute_quaternion_with_correction(current_quaternion, angular_velocity, bias, dt=dt, disturbance=0.0):
    angular_velocity_corrected = angular_velocity - bias - disturbance
    theta = torch.norm(angular_velocity_corrected, p=2, dim=-1, keepdim=True) * dt
    half_theta = theta / 2
    w = torch.cos(half_theta)
    xyz = torch.sin(half_theta) * angular_velocity_corrected / torch.norm(angular_velocity_corrected, p=2, dim=-1, keepdim=True)
    rotation_correction = torch.cat((w, xyz), dim=-1)

    # Normalize quaternions
    rotation_correction = rotation_correction / torch.norm(rotation_correction, p=2, dim=-1, keepdim=True)
    current_quaternion = current_quaternion / torch.norm(current_quaternion, p=2, dim=-1, keepdim=True)

    # Quaternion multiplication
    w1, x1, y1, z1 = current_quaternion.unbind(dim=-1)
    w2, x2, y2, z2 = rotation_correction.unbind(dim=-1)

    new_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    new_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    new_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    new_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    updated_rotation = torch.stack((new_w, new_x, new_y, new_z), dim=-1).float()  # Ensure dtype is float
    return updated_rotation

# Normalize disturbance (if needed, set to zero for simplicity)
disturbance = torch.zeros(3, device=device)

# Function to denormalize angular velocity
def denormalize_angular_velocity(angular_velocity, mean, std):
    mean = mean[3:6]
    std = std[3:6]
    return angular_velocity * std + mean

def quaternion_multiply(q, r):
    """
    Multiply two quaternions.
    """
    w0, x0, y0, z0 = q.unbind(-1)
    w1, x1, y1, z1 = r.unbind(-1)
    return torch.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    ], dim=-1)

def quaternion_conjugate(q):
    """
    Return the conjugate of a quaternion.
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quaternion_log(q):
    """
    Compute the logarithm of a quaternion.
    """
    q = q / q.norm(dim=-1, keepdim=True)  # Normalize the quaternion
    w, v = q[..., 0], q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True)
    v_norm = torch.where(v_norm > 1e-6, v_norm, torch.ones_like(v_norm) * 1e-6)  # Avoid division by zero
    angle = 2.0 * torch.atan2(v_norm, w)
    return angle[..., None] * (v / v_norm)


class CustomQuaternionLoss(nn.Module):
    def __init__(self):
        super(CustomQuaternionLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(delta=1.0)  # Huber loss function with delta=1.0

    def forward(self, true_quaternions, predicted_biases, batch_X, quaternions_all, indices, sequence_length, dt=dt, disturbance=0.0):
        batch_size = true_quaternions.size(0)
        quaternion_losses = []

        for i in range(batch_size):
            #angular_velocity_normalized = batch_X[i, -1, 3:6]  # Last angular velocity in sequence (normalized)
            #angular_velocity = denormalize_angular_velocity(angular_velocity_normalized, X_mean, X_std)
            angular_velocity = batch_X[i, -1, 3:6]
            
            true_quaternion = true_quaternions[i]

            # Calculate the actual index in the original dataset
            actual_index = indices[i]

            # Get the initial quaternion for the sequence
            initial_index = max(actual_index - (sequence_length - 1), 0)
            initial_quaternion = torch.tensor(quaternions_all[initial_index], dtype=torch.float32, device=batch_X.device)
            #initial_quaternion = quaternions_all[initial_index].clone().detach().to(dtype=torch.float32, device=batch_X.device)

            computed_quaternion = compute_quaternion_with_correction(initial_quaternion, angular_velocity, predicted_biases[i], dt=dt, disturbance=disturbance)

            # Compute quaternion error using log map on SO(3) space
            rotation_diff = quaternion_multiply(quaternion_conjugate(computed_quaternion), true_quaternion)
            log_rotation_diff = quaternion_log(rotation_diff)

            # Ensure log_rotation_diff requires grad
            #log_rotation_diff = log_rotation_diff.clone().detach().requires_grad_(True)

            quaternion_loss = self.huber_loss(log_rotation_diff, torch.zeros_like(log_rotation_diff, device=batch_X.device))
            quaternion_losses.append(quaternion_loss)

        if quaternion_losses:  # Check if quaternion_losses is not empty
            quaternion_losses = torch.stack(quaternion_losses)
            loss_quaternion = quaternion_losses.mean()
        else:
            print("Warning: quaternion_losses is empty.")
            loss_quaternion = torch.tensor(0.0, dtype=torch.float32, device=batch_X.device)

        return loss_quaternion
    
criterion = CustomQuaternionLoss()
#criterion = nn.MSELoss()

# K-Fold Cross Validation
k_folds = 20
kf = KFold(n_splits=k_folds, shuffle=True)

print('----Start Training (Gyro)----')



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
            true_quaternions = torch.tensor(quaternions_train[start_idx:end_idx], dtype=torch.float32, device=device)
            loss = criterion(true_quaternions, outputs, batch_X, quaternions_all, train_idx[start_idx:end_idx], sequence_length)
            
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
                true_quaternions = torch.tensor(quaternions_val[start_idx:end_idx], dtype=torch.float32, device=device)
                loss_quaternion = criterion(true_quaternions, outputs, batch_X, quaternions_all, val_idx[start_idx:end_idx], sequence_length)
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


train_dataset = GyroBiasDataset(X_train, y_train, biases_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GyroBiasDataset(X_val, y_val, biases_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_index, (batch_X, batch_y, batch_biases) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=True)):
        batch_X, batch_y, batch_biases = batch_X.to(device), batch_y.to(device), batch_biases.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X, batch_biases)
        
        denormalized_outputs = denormalize_model_output_iqr(outputs, bias_median, bias_iqr)

        start_idx = batch_index * train_dataloader.batch_size
        end_idx = start_idx + batch_X.size(0)
        #true_quaternions = torch.tensor(quaternions_train[start_idx:end_idx], dtype=torch.float32, device=device)
        #true_quaternions = quaternions_train[start_idx:end_idx].clone().detach().to(dtype=torch.float32, device=device)
        true_quaternions = quaternions_train[start_idx:end_idx].to(dtype=torch.float32, device=device)
        
        loss = criterion(true_quaternions, denormalized_outputs, batch_X, quaternions_all, train_idx[start_idx:end_idx], sequence_length)

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
            denormalized_outputs = denormalize_model_output_iqr(outputs, bias_median, bias_iqr)
            start_idx = batch_index * val_dataloader.batch_size
            end_idx = start_idx + batch_X.size(0)
            #true_quaternions = torch.tensor(quaternions_val[start_idx:end_idx], dtype=torch.float32, device=device)
            #true_quaternions = quaternions_val[start_idx:end_idx].clone().detach().to(dtype=torch.float32, device=device)
            true_quaternions = quaternions_val[start_idx:end_idx].to(dtype=torch.float32, device=device)
            
            loss_quaternion = criterion(true_quaternions, denormalized_outputs, batch_X, quaternions_all, val_idx[start_idx:end_idx], sequence_length)
            val_loss += loss_quaternion.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss:.20f}')

fold_results[0] = avg_val_loss


print('----Finished Training----')

# Save the trained model
model.to('cpu')

scripted_model = torch.jit.script(model)
if dataset == 'euroc':
    model_path = "euroc_scripted_gyro_model_10.pth"
elif dataset == 'kitti':
    model_path = "kitti_scripted_gyro_model_10.pth"
#torch.save(model.state_dict(), model_path)
scripted_model.save(model_path)

print(f"Model saved to {model_path}")

# Print fold results
for fold, loss in fold_results.items():
    print(f'Fold {fold+1}: Validation Loss = {loss}')

