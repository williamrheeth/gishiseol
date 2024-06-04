import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import KFold
import torch.nn.utils as nn_utils

# Cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# List of paths to your datasets
train_val_data_paths = [
    #'./ground_truth_estimate/data_aug_short.csv'
    './ground_truth_estimate/data_aug_mh01.csv',
    './ground_truth_estimate/data_aug_mh02.csv',
    './ground_truth_estimate/data_aug_mh03.csv',
    './ground_truth_estimate/data_aug_mh05.csv'
]
test_data_path = './ground_truth_estimate/data_aug_mh04.csv'
#test_data_path = './ground_truth_estimate/data_aug_short.csv'

# Define sequence length
sequence_length = 10

# Initialize lists to store sequences, and quaternions from all datasets
all_sequences = []
all_quaternions = []

# Normalize input feature values
def normalize(data):
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True)
    return (data - mean) / std, mean, std

def denormalize(data, mean, std):
    return data * std + mean


print('----Organizing Data----')

# Function to process dataset
def process_dataset(data_path):
    data = pd.read_csv(data_path)

    # Filter rows without NaNs in any of the specified columns
    filtered_data = data[~data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
                                'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].isnull().any(axis=1)]

    sequences = []
    quaternions = []
    for i in range(len(filtered_data) - sequence_length):
        sequence = filtered_data.iloc[i:i+sequence_length][[
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
            'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]'
        ]].values
        sequences.append(sequence)
        quaternion = filtered_data.iloc[i + sequence_length][[
            ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'
        ]].values
        quaternions.append(quaternion)

    sequences = np.array(sequences)
    quaternions = np.array(quaternions)

    return sequences, quaternions

# Process each dataset separately
for data_path in train_val_data_paths:
    X, quaternions = process_dataset(data_path)
    
    # Normalize input features for this dataset
    X, X_mean, X_std = normalize(X)
    
    # Append to the global lists
    all_sequences.append(X)
    all_quaternions.append(quaternions)

# Concatenate all sequences and quaternions
X_all = np.concatenate(all_sequences, axis=0)
y_all = np.concatenate(all_quaternions, axis=0)

# Process test dataset
X_test, y_test = process_dataset(test_data_path)
X_test, X_mean, X_std = normalize(X_test)

# Convert to PyTorch tensors
X_all = torch.tensor(X_all, dtype=torch.float32, device=device)
y_all = torch.tensor(y_all, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
X_mean = torch.tensor(X_mean, dtype=torch.float32, device=device)
X_std = torch.tensor(X_std, dtype=torch.float32, device=device)


print('\n----Organizing Complete!----\nsize: ')
print('X_all:', X_all.shape)
print('y_all:', y_all.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)



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
output_dim = 4  # Quaternion
num_epochs = 15 # Number of epochs
lr = 0.1      # Learning rate
batch_size = 256
# Scheduler hyperparamter
step_size = 5
gamma = 0.1
 


model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)



# Train
class GyroQuatDataset(Dataset):
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

    def forward(self, true_quaternions, predicted_quaternion, batch_X):
        batch_size = true_quaternions.size(0)
        quaternion_losses = []

        for i in range(batch_size):            
            #print(len(true_quaternions), len(predicted_quaternion))
            true_quaternion = true_quaternions[i]
            computed_quaternion = predicted_quaternion[i]
            #print(true_quaternion, computed_quaternion)
            
            if torch.isnan(true_quaternion).any():  # Check for any nan values        
                continue  # Skip the current iteration
            
            #if torch.norm(true_quaternion) < 1e-6:
            #   true_quaternion = self.eps * torch.ones_like(true_quaternion, device=device)* 1e-6
            
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
k_folds = 2
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
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        train_dataset = GyroQuatDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = GyroQuatDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(batch_y, outputs, batch_X)
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_dataloader)
            scheduler.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.14f}, LR: {scheduler.get_last_lr()[0]:.6f}')

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(batch_y, outputs, batch_X)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f'Validation Loss: {avg_val_loss:.20f}')

        fold_results[fold] = avg_val_loss

    print('----Finished Training----')

    # Save the trained model
    model_path = "gyro_end_to_end.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Print fold results
    for fold, loss in fold_results.items():
        print(f'Fold {fold+1}: Validation Loss = {loss}')
else:
    print('Loading Previously saved Model')
    model_path = "gyro_end_to_end.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    

    
# After training, evaluate on the test set
test_dataset = GyroQuatDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()

X_test = X_test.to(device)
disturbance = disturbance.to(device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

with torch.no_grad():
    outputs = model(X_test)

# Compute quaternion errors for the test set
initial_quaternion = y_test[0]  # Starting quaternion from the test set

total_quaternion_error = 0

for i in range(len(X_test)):
    
    true_quaternion = y_test[i]
    predicted_quaternion = outputs[i]

    # Normalize quaternions
    true_quaternion = true_quaternion / torch.norm(true_quaternion, p=2, dim=-1, keepdim=True)
    predicted_quaternion = predicted_quaternion / torch.norm(predicted_quaternion, p=2, dim=-1, keepdim=True)

    quaternion_error = mean_squared_error(true_quaternion.cpu().numpy(), predicted_quaternion.cpu().numpy())
    total_quaternion_error += quaternion_error

    if i < 30:
        print(f'Sequence {i+1}:')
        print(f'  Predicted Quaternion: {predicted_quaternion}')
        print(f'  True Quaternion: {true_quaternion}')
        print(f'  Quaternion Error: {quaternion_error}')

# Calculate average quaternion error
net_average_quaternion_error = total_quaternion_error / len(X_test)

print(f'Net Average Quaternion Error: {net_average_quaternion_error}')