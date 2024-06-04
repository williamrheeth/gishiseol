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
    #'./ground_truth_estimate/data_aug_short.csv'
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
#X_mean = torch.tensor(X_mean, dtype=torch.float32, device=device)
#X_std = torch.tensor(X_std, dtype=torch.float32, device=device)
#bias_mean = torch.tensor(bias_mean, dtype=torch.float32, device=device)
#bias_std = torch.tensor(bias_std, dtype=torch.float32, device=device)

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
#
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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Function to compute quaternion using the given equation# Function to compute quaternion using the given equation
def compute_quaternion_with_correction(current_quaternion, angular_velocity, bias, dt=0.01, disturbance=0.0):
    angular_velocity_corrected = angular_velocity - bias - disturbance
    rotation_vector = angular_velocity_corrected * dt
    rotation_correction = R.from_rotvec(rotation_vector).as_quat()  # Corrected rotation as quaternion
    current_rotation = R.from_quat(current_quaternion)  # Current orientation as Rotation object
    updated_rotation = current_rotation * R.from_quat(rotation_correction)  # Apply correction
    updated_quaternion = updated_rotation.as_quat()  # Get updated quaternion
    return updated_quaternion

# Normalize disturbance (if needed, set to zero for simplicity)
disturbance = np.zeros(3)

# Function to denormalize angular velocity
def denormalize_angular_velocity(angular_velocity, mean, std):
    mean = mean[0, 0, 3:6]  # Extract mean for angular velocity
    std = std[0, 0, 3:6]    # Extract std for angular velocity
    return angular_velocity * std + mean

alpha = 0
print(f'\n#############################\n       Alpha = {alpha}\n#############################\n')

print('----Start Training----')

# K-Fold Cross Validation
k_folds = 8
kf = KFold(n_splits=k_folds, shuffle=True)

trained = False

# Training loop
if trained == False:
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Transfer data to GPU
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
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
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f'Validation Loss: {avg_val_loss:.4f}')
            
        fold_results[fold] = avg_val_loss
    
    # Save the trained model
    model_path = f"gyro_bias_transformer_model_bias_mse.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
else:
    print('Loading Previously saved Model')
    model_path = f"gyro_bias_transformer_model_bias_mse.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    
print('----Finished Training----')
    
# After training, evaluate on the test set
test_dataset = GyroBiasDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
test_loss = 0
predicted_biases_all = []
with torch.no_grad():
    for batch_X, batch_y in test_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss_bias = criterion(outputs, batch_y)
        test_loss += loss_bias.item()
        predicted_biases_all.append(outputs.cpu().numpy())
avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss: {avg_test_loss:.4f}')

# Concatenate all predicted biases
predicted_biases = np.concatenate(predicted_biases_all, axis=0)

# Denormalize predicted biases
predicted_biases = denormalize_bias(predicted_biases, bias_mean, bias_std)

# Compare the computed quaternion to the ground truth quaternion for test set
initial_quaternion = quaternions_test[0]  # Starting quaternion from the test set

total_bias_mse = 0
total_quaternion_error = 0

for i in range(len(X_test)):
    angular_velocity_normalized = X_test[i, -1, 3:6].cpu().numpy()  # Last angular velocity in sequence (normalized)
    angular_velocity = denormalize_angular_velocity(angular_velocity_normalized, X_mean, X_std)
    true_quaternion = quaternions_test[i]
    predicted_bias = predicted_biases[i]
    current_quaternion = initial_quaternion if i == 0 else computed_quaternion
    computed_quaternion = compute_quaternion_with_correction(current_quaternion, angular_velocity, predicted_bias, dt=0.01, disturbance=disturbance)
    
    bias_mse = mean_squared_error(y_test[i].cpu().numpy(), predicted_bias)
    quaternion_error = mean_squared_error(true_quaternion, computed_quaternion)
    
    total_bias_mse += bias_mse
    total_quaternion_error += quaternion_error
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