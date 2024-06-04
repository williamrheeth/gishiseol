import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import KFold

# Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# List of paths to your datasets
train_val_data_paths = [
    './ground_truth_estimate/data_aug_mh01.csv',
    './ground_truth_estimate/data_aug_mh02.csv',
    './ground_truth_estimate/data_aug_mh03.csv',
    './ground_truth_estimate/data_aug_mh05.csv'
]
test_data_path = './ground_truth_estimate/data_aug_mh04.csv'

# Define sequence length
sequence_length = 10

# Initialize lists to store sequences, biases, and quaternions from all datasets
all_sequences = []
all_biases = []
all_quaternions = []

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
    quaternions = []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length][[
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
            'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
            ' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]'
        ]].values
        sequences.append(sequence)
        quaternion = data.iloc[i + sequence_length][[
            ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'
        ]].values
        quaternions.append(quaternion)
    
    sequences = np.array(sequences)
    quaternions = np.array(quaternions)
    X = sequences[:, :, :-3]  # Acceleration, angular velocity, previous gyro bias
    y = sequences[:, -1, -3:]  # Gyro bias
    
    return X, y, quaternions

# Process each dataset separately
for data_path in train_val_data_paths:
    X, y, quaternions = process_dataset(data_path)
    
    # Normalize input features and bias values for this dataset
    X, X_mean, X_std = normalize(X)
    y, bias_mean, bias_std = normalize_bias(y)
    
    # Append to the global lists
    all_sequences.append(X)
    all_biases.append(y)
    all_quaternions.append(quaternions)

# Concatenate all sequences, biases, and quaternions
X_all = np.concatenate(all_sequences, axis=0)
y_all = np.concatenate(all_biases, axis=0)
quaternions_all = np.concatenate(all_quaternions, axis=0)

# Process test dataset
X_test, y_test, quaternions_test = process_dataset(test_data_path)
X_test, X_mean, X_std = normalize(X_test)
y_test, bias_mean, bias_std = normalize_bias(y_test)

# Convert to PyTorch tensors
#X_all = torch.tensor(X_all, dtype=torch.float32)
#y_all = torch.tensor(y_all, dtype=torch.float32)
#X_test = torch.tensor(X_test, dtype=torch.float32)
#y_test = torch.tensor(y_test, dtype=torch.float32)
X_all = torch.tensor(X_all, dtype=torch.float32, device=device)
y_all = torch.tensor(y_all, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
X_mean = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_std = torch.tensor(X_std, dtype=torch.float32, device=device)
bias_mean = torch.tensor(bias_mean, dtype=torch.float32, device=device)
bias_std = torch.tensor(bias_std, dtype=torch.float32, device=device)

print('\n----Organizing Complete!----\nsize: ')
print('X_all:', X_all.shape)
print('y_all:', y_all.shape)
print('quaternions_all:', quaternions_all.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)
print('quaternions_test:', quaternions_test.shape)



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
#z
#                                       HYPERPARAMETER                                       #                               
#
##############################################################################################
# Model hyperparameters
input_dim = 6   # 3 (acc) + 3 (gyro)
model_dim = 64  # Model dimension
num_heads = 8   # Number of heads (# of self-attention)
num_layers = 6  # Number of encoder-decoder layers (cross-attention)
output_dim = 3  # Gyro bias (3D)
num_epochs = 15 # Number of epochs
lr = 0.1      # Learning rate
batch_size = 256
# Scheduler hyperparamter
step_size = 5
gamma = 0.1
 


model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)



# Train
class GyroBiasDataset(Dataset):
    def __init__(self, X, y):
        #self.X = X
        #self.y = y
        self.X = X.to(device)
        self.y = y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Function to compute quaternion using the given equation
def compute_quaternion_with_correction(current_quaternion, angular_velocity, bias, dt=0.01, disturbance=0.0):
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
    #mean = torch.tensor(mean[0, 0, 3:6], device=angular_velocity.device, dtype=torch.float32)  # Extract mean for angular velocity
    #std = torch.tensor(std[0, 0, 3:6], device=angular_velocity.device, dtype=torch.float32)    # Extract std for angular velocity
    mean = mean[0, 0, 3:6]
    std = std[0, 0, 3:6]
    return angular_velocity * std + mean


class CustomQuaternionLoss(nn.Module):
    def __init__(self):
        super(CustomQuaternionLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(delta=1.0)  # Huber loss function with delta=1.0

    def forward(self, true_quaternions, predicted_biases, batch_X, quaternions_all, indices, sequence_length, dt=0.01, disturbance=0.0):
        batch_size = true_quaternions.size(0)
        quaternion_losses = []

        for i in range(batch_size):
            angular_velocity_normalized = batch_X[i, -1, 3:6]  # Last angular velocity in sequence (normalized)
            #angular_velocity = denormalize_angular_velocity(angular_velocity_normalized, X_mean, X_std).to(batch_X.device).float()
            #true_quaternion = true_quaternions[i].float()
            angular_velocity = denormalize_angular_velocity(angular_velocity_normalized, X_mean, X_std)
            true_quaternion = true_quaternions[i]

            # Calculate the actual index in the original dataset
            actual_index = indices[i]

            # Get the initial quaternion for the sequence
            initial_index = max(actual_index - (sequence_length - 1), 0)
            initial_quaternion = torch.tensor(quaternions_all[initial_index], dtype=torch.float32, device=batch_X.device)

            computed_quaternion = compute_quaternion_with_correction(initial_quaternion, angular_velocity, predicted_biases[i], dt=dt, disturbance=disturbance)
            
            # Compute quaternion error using log map on SO(3) space
            rotation_diff = R.from_quat(computed_quaternion.detach().cpu().numpy()) * R.from_quat(true_quaternion.detach().cpu().numpy()).inv()
            log_rotation_diff = rotation_diff.as_rotvec()
            log_rotation_diff = torch.tensor(log_rotation_diff, dtype=torch.float32, device=batch_X.device)

            # Ensure log_rotation_diff requires grad
            log_rotation_diff.requires_grad = True

            #quaternion_loss = self.huber_loss(log_rotation_diff, torch.zeros_like(log_rotation_diff))
            #quaternion_losses.append(quaternion_loss)        
            log_rotation_diff = log_rotation_diff.to(device)  # Move log_rotation_diff to GPU
            quaternion_loss = self.huber_loss(log_rotation_diff, torch.zeros_like(log_rotation_diff, device=device))
            quaternion_losses.append(quaternion_loss)

        if quaternion_losses:  # Check if quaternion_losses is not empty
            #quaternion_losses = torch.stack(quaternion_losses).float()  # Ensure dtype is float
            quaternion_losses = torch.stack(quaternion_losses)
            loss_quaternion = quaternion_losses.mean()
        else:
            print("Warning: quaternion_losses is empty.")
            loss_quaternion = torch.tensor(0.0, dtype=torch.float32, device=batch_X.device)

        return loss_quaternion
    
criterion = CustomQuaternionLoss()




# K-Fold Cross Validation
k_folds = 8
kf = KFold(n_splits=k_folds, shuffle=True)

print('----Start Training----')
trained = True

if not trained:
    fold_results = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f'Fold {fold+1}/{k_folds}')

        # Split data
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        quaternions_train = quaternions_all[train_idx]
        quaternions_val = quaternions_all[val_idx]

        train_dataset = GyroBiasDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = GyroBiasDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_index, (batch_X, batch_y) in enumerate(train_dataloader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                
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
                for batch_index, (batch_X, batch_y) in enumerate(val_dataloader):
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    start_idx = batch_index * val_dataloader.batch_size
                    end_idx = start_idx + batch_X.size(0)
                    true_quaternions = torch.tensor(quaternions_val[start_idx:end_idx], dtype=torch.float32, device=device)
                    loss_quaternion = criterion(true_quaternions, outputs, batch_X, quaternions_all, val_idx[start_idx:end_idx], sequence_length)
                    val_loss += loss_quaternion.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f'Validation Loss: {avg_val_loss:.20f}')

        fold_results[fold] = avg_val_loss

    print('----Finished Training----')

    # Save the trained model
    model_path = "gyro_bias_transformer_only_quat_model_kfold.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Print fold results
    for fold, loss in fold_results.items():
        print(f'Fold {fold+1}: Validation Loss = {loss}')
else:
    print('Loading Previously saved Model')
    model_path = "gyro_bias_transformer_only_quat_model_kfold.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    

    
# After training, evaluate on the test set
test_dataset = GyroBiasDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
'''
test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        start_idx = batch_index * test_dataloader.batch_size
        end_idx = start_idx + batch_X.size(0)
        true_quaternions = torch.tensor(quaternions_test[start_idx:end_idx], dtype=torch.float32, device=device)
        loss_bias = criterion(true_quaternions, outputs, batch_X, quaternions_all, test_indices[start_idx:end_idx], sequence_length)
        test_loss += loss_bias.item()
avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss: {avg_test_loss:.4f}')
'''

X_test = X_test.to(device)
disturbance = disturbance.to(device)
quaternions_test = torch.tensor(quaternions_test, dtype=torch.float32, device=device)

# Denormalize predicted biases
with torch.no_grad():
    outputs = model(X_test)
#predicted_biases = denormalize_bias(outputs.cpu().numpy(), bias_mean, bias_std).to(device)
predicted_biases = denormalize_bias(outputs, bias_mean, bias_std)

# Compare the computed quaternion to the ground truth quaternion for test set
initial_quaternion = quaternions_test[0]  # Starting quaternion from the test set

total_bias_mse = 0
total_quaternion_error = 0

def denormalize_angular_velocity1(angular_velocity, mean, std):
    mean = mean[0, 0, 3:6]  # Extract mean for angular velocity
    std = std[0, 0, 3:6]    # Extract std for angular velocity
    return angular_velocity * std + mean

def compute_quaternion_with_correction1(current_quaternion, angular_velocity, bias, dt=0.01, disturbance=0.0):
    # Convert disturbance to numpy array if it is a PyTorch tensor
    #if isinstance(disturbance, torch.Tensor):
    #    disturbance = disturbance.numpy().cpu()

    # Perform the subtraction operation
    angular_velocity_corrected = angular_velocity - bias
    rotation_vector = angular_velocity_corrected.cpu().numpy() * dt  # Move angular velocity to CPU before converting to NumPy
    rotation_correction = R.from_rotvec(rotation_vector).as_quat()  # Corrected rotation as quaternion
    current_rotation = R.from_quat(current_quaternion.cpu().numpy())  # Current orientation as Rotation object, move to CPU
    updated_rotation = current_rotation * R.from_quat(rotation_correction)  # Apply correction
    updated_quaternion = updated_rotation.as_quat()  # Get updated quaternion
    return torch.from_numpy(updated_quaternion)  # Convert back to PyTorch tensor

for i in range(len(X_test)):
    #angular_velocity_normalized = X_test[i, -1, 3:6].cpu().numpy()  # Last angular velocity in sequence (normalized)
    angular_velocity_normalized = X_test[i, -1, 3:6]
    angular_velocity = denormalize_angular_velocity1(angular_velocity_normalized, X_mean, X_std)
    true_quaternion = quaternions_test[i]
    predicted_bias = predicted_biases[i]
    current_quaternion = initial_quaternion if i == 0 else computed_quaternion
    computed_quaternion = compute_quaternion_with_correction1(current_quaternion, angular_velocity, predicted_bias, dt=0.01, disturbance=disturbance)
    #computed_quaternion = compute_quaternion_with_correction1(current_quaternion, angular_velocity, predicted_bias, dt=0.01, disturbance=disturbance.to(device))
    
    #bias_mse = mean_squared_error(y_test[i].cpu().numpy(), predicted_bias)
    #quaternion_error = mean_squared_error(true_quaternion, computed_quaternion)
    bias_mse = mean_squared_error(y_test[i].cpu().numpy(), predicted_bias.cpu().numpy())
    quaternion_error = mean_squared_error(true_quaternion.cpu().numpy(), computed_quaternion.cpu().numpy())
    
    #total_bias_mse += bias_mse
    #total_quaternion_error += quaternion_error
    total_bias_mse += bias_mse.item()
    total_quaternion_error += quaternion_error.item()
    
    
    if i < 30:
        print(f'Sequence {i+1}:')
        print(f'  Predicted Bias: {predicted_bias}')
        print(f'  True Bias: {y_test[i].cpu().numpy()}')
        print(f'  True Quaternion: {true_quaternion}')
        print(f'  Computed Quaternion: {computed_quaternion}')
        print(f'  MSE Bias: {bias_mse}')
        print(f'  Quaternion Error: {quaternion_error}')

    # Update initial_quaternion for the next iteration
    initial_quaternion = computed_quaternion
    
# Calculate average errors
net_average_bias_mse = total_bias_mse / len(X_test)
net_average_quaternion_error = total_quaternion_error / len(X_test)

print(f'Net Average Bias MSE: {net_average_bias_mse}')
print(f'Net Average Quaternion Error: {net_average_quaternion_error}')