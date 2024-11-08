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

from utils import (
    normalize, 
    denormalize, 
    quaternion_product, 
    quaternion_conjugate, 
    quaternion_log, 
    denormalize_model_output_iqr, 
    compute_quaternion_with_correction, 
    compute_acceleration_with_correction, 
    quaternion_rotate
)

def main(): 
        
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
            './transformer_v4/augmented_data/mh01.csv',
            './transformer_v4/augmented_data/mh03.csv',
            './transformer_v4/augmented_data/mh05.csv',
            
            './transformer_v4/augmented_data/v102.csv',
            './transformer_v4/augmented_data/v201.csv',
            './transformer_v4/augmented_data/v203.csv',
            
            './transformer_v4/augmented_data/mh01_truebias.csv',
            './transformer_v4/augmented_data/mh03_truebias.csv',
            './transformer_v4/augmented_data/mh05_truebias.csv',
            
            './transformer_v4/augmented_data/v102_truebias.csv',
            './transformer_v4/augmented_data/v201_truebias.csv',
            './transformer_v4/augmented_data/v203_truebias.csv',
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
    imu_sequence_length = 100
    bias_sequence_length = 10


    # Directory to save model checkpoints and loss logs
    save_dir = "./transformer_v4/checkpoints"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    # File to save loss logs
    loss_log_file = os.path.join(save_dir, "loss_log.txt")



    # Initialize lists to store sequences, biases, and quaternions from all datasets
    all_imu_data = []
    all_biases = []
    all_initial_quaternions = []
    all_quaternions = []
    all_initial_positions = []
    all_positions = []
    all_initial_velocities = []
    all_velocities = []

    print('----Organizing Data----')

    # Function to process dataset
    def process_dataset(data_path):
        data = pd.read_csv(data_path)
        
        imu_datas = []
        biases = []
        initial_quaternions = []
        quaternions = []
        initial_positions = []
        positions = []
        initial_velocities = []
        velocities = []
        
        for i in range(len(data) - imu_sequence_length - 1):
            imu_data = data.iloc[i:i+imu_sequence_length][[
                'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
                'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]'
            ]].values
            imu_datas.append(imu_data)
            
            bias = data.iloc[i+imu_sequence_length-bias_sequence_length:i+imu_sequence_length][[   # Last 20 values
                #' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]'
                'b_a_RS_S_x [m s^-2]', 'b_a_RS_S_y [m s^-2]', 'b_a_RS_S_z [m s^-2]',
                'b_w_RS_S_x [rad s^-1]', 'b_w_RS_S_y [rad s^-1]', 'b_w_RS_S_z [rad s^-1]'
            ]].values
            biases.append(bias)
            
            initial_quaternion = data.iloc[i+imu_sequence_length-bias_sequence_length:i+imu_sequence_length][[
                #'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
                'true_q_w[]', 'true_q_x[]', 'true_q_y[]', 'true_q_z[]'
            ]].values
            initial_quaternions.append(initial_quaternion)
            
            quaternion = data.iloc[i+imu_sequence_length-bias_sequence_length+1:i+imu_sequence_length+1][[
                'true_q_w[]', 'true_q_x[]', 'true_q_y[]', 'true_q_z[]'
            ]].values
            quaternions.append(quaternion)

            initial_position = data.iloc[i+imu_sequence_length-bias_sequence_length:i+imu_sequence_length][[
                #'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]'
                'true_p_x[m]', 'true_p_y[m]', 'true_p_z[m]'
            ]].values
            initial_positions.append(initial_position)

            position = data.iloc[i+imu_sequence_length-bias_sequence_length+1:i+imu_sequence_length+1][[
                'true_p_x[m]', 'true_p_y[m]', 'true_p_z[m]'
            ]].values
            positions.append(position)

            initial_velocity = data.iloc[i+imu_sequence_length-bias_sequence_length:i+imu_sequence_length][[
                'v_RS_R_x [m s^-1]', 'v_RS_R_y [m s^-1]', 'v_RS_R_z [m s^-1]'
            ]].values
            initial_velocities.append(initial_velocity)
            
            velocity = data.iloc[i+imu_sequence_length-bias_sequence_length+1:i+imu_sequence_length+1][[
                'v_RS_R_x [m s^-1]', 'v_RS_R_y [m s^-1]', 'v_RS_R_z [m s^-1]'
            ]].values
            velocities.append(velocity)
        
        imu_datas = np.array(imu_datas)
        biases = np.array(biases)
        initial_quaternions = np.array(initial_quaternions)
        quaternions = np.array(quaternions)
        initial_positions = np.array(initial_positions)
        positions = np.array(positions)
        initial_velocities = np.array(initial_velocities)
        velocities = np.array(velocities)
        
        return imu_datas, biases, initial_quaternions, quaternions, initial_positions, positions, initial_velocities, velocities



    # Process each dataset separately
    for data_path in train_val_data_paths:
        # Extract dataset name from file path for saving/loading and append "accel"
        dataset_name = os.path.basename(data_path).split('.')[0]  # e.g., 'data_aug_mh01'
        processed_data_path = f'./transformer_v4/augmented_data/processed_{dataset_name}.pth'

        if os.path.exists(processed_data_path):
            print(f'----Loading Preprocessed Data for {dataset_name}----')
            # Load preprocessed data
            checkpoint = torch.load(processed_data_path, map_location=device)

            imu_datas = checkpoint['imu_datas']
            biases = checkpoint['biases']
            initial_quaternions = checkpoint['initial_quaternions']
            quaternions = checkpoint['quaternions']
            initial_positions = checkpoint['initial_positions']
            positions = checkpoint['positions']
            initial_velocities = checkpoint['initial_velocities']
            velocities = checkpoint['velocities']

        else:
            print(f'----Processing Data for {dataset_name}----')
            imu_datas, biases, initial_quaternions, quaternions, initial_positions, positions, initial_velocities, velocities = process_dataset(data_path)

            # Save the processed data
            print(f'----Saving Preprocessed Data for {dataset_name}----')
            torch.save({
                'imu_datas': imu_datas,
                'biases': biases,
                'initial_quaternions': initial_quaternions,
                'quaternions': quaternions,
                'initial_positions': initial_positions,
                'positions': positions,
                'initial_velocities': initial_velocities,
                'velocities': velocities,
            }, processed_data_path)

        # Append to the global lists
        all_imu_data.append(imu_datas)
        all_biases.append(biases)
        all_initial_quaternions.append(initial_quaternions)
        all_quaternions.append(quaternions)
        all_initial_positions.append(initial_positions)
        all_positions.append(positions)
        all_initial_velocities.append(initial_velocities)
        all_velocities.append(velocities)

    # Concatenate all sequences, biases, and quaternions
    X_all = np.concatenate(all_imu_data, axis=0)
    biases_all = np.concatenate(all_biases, axis=0)
    initial_quaternions_all = np.concatenate(all_initial_quaternions, axis=0)
    quaternions_all = np.concatenate(all_quaternions, axis=0)
    initial_positions_all = np.concatenate(all_initial_positions, axis=0)
    positions_all = np.concatenate(all_positions, axis=0)
    initial_velocities_all = np.concatenate(all_initial_velocities, axis=0)
    velocities_all = np.concatenate(all_velocities, axis=0)


    '''
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
    '''

    X_mean = X_all.mean(axis=(0, 1))
    X_std = X_all.std(axis=(0, 1))

    bias_mean = biases_all.mean(axis=(0, 1))
    bias_std = biases_all.std(axis=(0, 1))


    # Convert to PyTorch tensors
    X_all = torch.tensor(X_all, dtype=torch.float32, device=device)
    biases_all = torch.tensor(biases_all, dtype=torch.float32, device=device)

    initial_quaternions_all = torch.tensor(initial_quaternions_all, dtype=torch.float32, device=device)
    quaternions_all = torch.tensor(quaternions_all, dtype=torch.float32, device=device)
    initial_positions_all = torch.tensor(initial_positions_all, dtype=torch.float32, device=device)
    positions_all = torch.tensor(positions_all, dtype=torch.float32, device=device)
    initial_velocities_all = torch.tensor(initial_velocities_all, dtype=torch.float32, device=device)
    velocities_all = torch.tensor(velocities_all, dtype=torch.float32, device=device)


    print('\n----Organizing Complete!----\nsize: ')
    print('X_all:', X_all.shape)
    print('biases_all:', biases_all.shape)
    print('initial_quaternions_all:', initial_quaternions_all.shape)
    print('initial_velocities_all:', initial_velocities_all.shape)
    print('initial_positions_all:', initial_positions_all.shape)
    print('quaternions_all:', quaternions_all.shape)
    print('velocities_all:', velocities_all.shape)
    print('positions_all:', positions_all.shape)
    print('\nX_mean:', X_mean)
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
        def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, seq_length, dropout=0.1):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Linear(input_dim, model_dim)
            self.decoder_embedding = nn.Linear(output_dim, model_dim)
            self.positional_encoding = PositionalEncoding(model_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
            decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
            self.fc = nn.Linear(model_dim, output_dim)
            self.seq_length = seq_length  # Number of time steps (10)

        def forward(self, src, tgt):
            src = self.embedding(src)
            src = self.positional_encoding(src)
            tgt = self.decoder_embedding(tgt)
            tgt = self.positional_encoding(tgt)
            memory = self.transformer_encoder(src)
            output = self.transformer_decoder(tgt, memory)
            output = self.fc(output)  # Shape: (batch_size, seq_length, output_dim * seq_length)
            # Reshape the output to (batch_size, seq_length, output_dim) to return the predictions for all 20 points
            #output = output.view(output.size(0), self.seq_length, -1)
            return output

    ############################################################################################################################
    #                                              Model hyperparameters                                                       #
    ############################################################################################################################

    input_dim = 6       # 3 (acc) + 3 (gyro)
    model_dim = 64     # Model dimension
    num_heads = 8       # Number of heads (for multi-head attention)
    num_layers = 2      # Number of encoder-decoder layers (cross-attention)
    output_dim = 6      # Gyro + Accel bias increment (3D for each)
    seq_length = 10     # 10 IMU frames in 1 camera frame
    dropout = 0.2       # Dropout prob
    num_epochs = 15     # Number of epochs
    lr = 1e-6           # Learning rate
    batch_size = 32     # Batch size
    step_size = 5       # Scheduler step size
    gamma = 0.1         # Scheduler gamma

    ############################################################################################################################

    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, seq_length, dropout).to(device)

    # Train
    class GyroBiasDataset(Dataset):
        def __init__(self, X, biases, init_quaternions, init_positions, init_velocities, quaternions, positions, velocities):
            self.X = X
            self.biases = biases
            self.init_quaternions = init_quaternions
            self.init_positions = init_positions
            self.init_velocities = init_velocities
            self.quaternions = quaternions
            self.positions = positions
            self.velocities = velocities

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return {
                'imu_data': self.X[idx],  # IMU data (input)
                'biases': self.biases[idx],  # Bias values (target)
                'init_quaternion': self.init_quaternions[idx],  # Initial quaternions
                'init_position': self.init_positions[idx],  # Initial positions
                'init_velocity': self.init_velocities[idx],  # Initial velocities
                'quaternion': self.quaternions[idx],  # True quaternions
                'position': self.positions[idx],  # True positions
                'velocity': self.velocities[idx],  # True velocities
            }

    # Normalize disturbance (if needed, set to zero for simplicity)
    disturbance = torch.zeros(3, device=device)

    class CombinedLoss(nn.Module):
        def __init__(self, position_weight=1.0, quaternion_weight=1.0, timestep_weighting='linear'):
            super(CombinedLoss, self).__init__()
            self.position_loss_fn = nn.HuberLoss(delta=1.0, reduction='none')  # Using reduction='none' for per-timestep loss
            self.quaternion_loss_fn = nn.HuberLoss(delta=1.0, reduction='none')
            self.position_weight = position_weight
            self.quaternion_weight = quaternion_weight
            self.timestep_weighting = timestep_weighting

        def get_timestep_weights(self, sequence_length):
            """ Calculate the weights for each timestep across the entire batch at once. """
            if self.timestep_weighting == 'linear':
                return torch.linspace(1 / sequence_length, 1, steps=sequence_length)
            elif self.timestep_weighting == 'exponential':
                return torch.exp(torch.linspace(0, 1, steps=sequence_length)) / sequence_length
            else:
                return torch.ones(sequence_length)

        def forward(self, true_quaternions, true_positions, predicted_biases, batch_X, init_positions, init_quaternions, init_velocities, gravity_vector, sequence_length, dt=0.005, disturbance=0.0):
            batch_size = true_positions.size(0)
            
            # Use only the initial values from the batch
            current_position = init_positions[:, 0].to(batch_X.device)
            current_velocity = init_velocities[:, 0].to(batch_X.device)
            current_quaternion = init_quaternions[:, 0].to(batch_X.device)

            # Split the predicted biases into accelerometer and gyroscope biases
            accel_bias = predicted_biases[:, :, 0:3]  # Accelerometer bias for each timestep
            gyro_bias = predicted_biases[:, :, 3:6]   # Gyroscope bias for each timestep

            # Prepare timestep weights for vectorized loss weighting
            timestep_weights = self.get_timestep_weights(sequence_length).to(batch_X.device)

            # Position and quaternion losses across timesteps
            position_losses = []
            quaternion_losses = []

            # Propagate across sequence timesteps in a vectorized manner
            for t in range(sequence_length):
                # Compute current timestep acceleration and angular velocity
                acceleration = batch_X[:, t, 0:3]  # Accelerometer data
                angular_velocity = batch_X[:, t, 3:6]  # Gyroscope data

                # Update position, velocity, and quaternion with correction
                current_position = compute_acceleration_with_correction(
                    current_position, acceleration, accel_bias[:, t], current_velocity, gravity_vector, current_quaternion, dt=dt, disturbance=disturbance
                )
                current_velocity = current_velocity + dt * (quaternion_rotate(current_quaternion, acceleration) + gravity_vector + disturbance)
                current_quaternion = compute_quaternion_with_correction(
                    current_quaternion, angular_velocity, gyro_bias[:, t], dt=dt, disturbance=disturbance
                )

                # Calculate loss for the current timestep
                true_position = true_positions[:, t]
                true_quaternion = true_quaternions[:, t]

                position_loss = self.position_loss_fn(current_position, true_position) * timestep_weights[t]
                quaternion_difference = quaternion_log(quaternion_product(quaternion_conjugate(current_quaternion), true_quaternion))
                quaternion_difference_4d = torch.cat((torch.zeros_like(quaternion_difference[..., :1]), quaternion_difference), dim=-1)
                quaternion_loss = self.quaternion_loss_fn(quaternion_difference_4d, torch.zeros_like(quaternion_difference_4d, device=batch_X.device)) * timestep_weights[t]

                position_losses.append(position_loss)
                quaternion_losses.append(quaternion_loss)

            # Stack losses, normalize by sequence mean, and combine with weights
            position_losses = torch.stack(position_losses, dim=0).mean(dim=0)
            quaternion_losses = torch.stack(quaternion_losses, dim=0).mean(dim=0)
            
            # Normalize to bring losses to similar magnitudes
            position_mean_loss = position_losses.mean()
            quaternion_mean_loss = quaternion_losses.mean()
            
            normalized_position_loss = position_losses / (position_mean_loss + 1e-6)
            normalized_quaternion_loss = quaternion_losses / (quaternion_mean_loss + 1e-6)

            # Weighted sum of normalized losses
            total_loss = self.position_weight * normalized_position_loss.mean() + \
                        self.quaternion_weight * normalized_quaternion_loss.mean()

            return total_loss
        
    criterion = CombinedLoss()

    # K-Fold Cross Validation
    k_folds = 20
    kf = KFold(n_splits=k_folds, shuffle=True)

    print('----Start Training (Accel)----')


    # Get the first train-test split
    train_idx, val_idx = next(kf.split(X_all))

    # Split data
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    biases_train, biases_val = biases_all[train_idx], biases_all[val_idx]
    init_quaternions_train, init_quaternions_val = quaternions_all[train_idx], quaternions_all[val_idx]
    init_positions_train, init_positions_val = positions_all[train_idx], positions_all[val_idx]
    init_velocities_train, init_velocities_val = velocities_all[train_idx], velocities_all[val_idx]
    true_quaternions_train, true_quaternions_val = quaternions_all[train_idx], quaternions_all[val_idx]
    true_positions_train, true_positions_val = positions_all[train_idx], positions_all[val_idx]
    velocities_train, velocities_val = velocities_all[train_idx], velocities_all[val_idx]


    # Gravity
    gravity_vector = torch.tensor([0, 0, -9.81], device=device)

    train_dataset = GyroBiasDataset(
        X_train, 
        biases_train, 
        init_quaternions_train, 
        init_positions_train, 
        init_velocities_train, 
        true_quaternions_train, 
        true_positions_train, 
        velocities_train
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = GyroBiasDataset(
        X_val, 
        biases_val, 
        init_quaternions_val, 
        init_positions_val, 
        init_velocities_val, 
        true_quaternions_val, 
        true_positions_val, 
        velocities_val
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    data_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_index, batch_data in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=True)):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}  # Move all data to GPU
            # Unpack the batch data
            imu_data = batch_data['imu_data']  # IMU data (input)
            biases = batch_data['biases']  # Biases (target)
            init_quaternions = batch_data['init_quaternion']  # Initial quaternions
            init_positions = batch_data['init_position']  # Initial positions
            init_velocities = batch_data['init_velocity']  # Initial velocities
            true_quaternions = batch_data['quaternion']  # True quaternions
            true_positions = batch_data['position']  # True positions
            true_velocities = batch_data['velocity']
            
            optimizer.zero_grad()
            
            # Model forward pass (predicts increment of bias)
            predicted_increments = model(imu_data, biases)

            # Calculate the final bias by adding the increment to the initial bias for each timestep
            predicted_biases = biases + predicted_increments  # Add increment to the biases at each timestep

            loss = criterion(
                true_quaternions=true_quaternions, 
                true_positions=true_positions, 
                predicted_biases=predicted_biases, 
                batch_X=imu_data, 
                init_positions=init_positions,  
                init_quaternions=init_quaternions, 
                init_velocities=init_velocities,  
                gravity_vector=gravity_vector, 
                sequence_length=seq_length
            )

            
            # Show estimated pos and quat
            if batch_index % 100 == 0:
                # Compute estimated positions using the predicted biases
                estimated_positions = []
                estimated_quaternions = []
                
                for i in range(imu_data.size(0)):
                    # Fetch initial position and velocity from training data
                    initial_position = init_positions[i].to(device)
                    initial_velocity = init_velocities[i].to(device)
                    initial_quaternion = init_quaternions[i].to(device)
                    
                    # Split the bias into accelerometer and gyroscope parts (3D each)
                    accel_bias = predicted_biases[i, :, 0:3]  # First 3 values for accelerometer bias
                    gyro_bias = predicted_biases[i, :, 3:6]   # Last 3 values for gyroscope bias
                    
                    # Compute the estimated position using the corrected acceleration
                    estimated_position = compute_acceleration_with_correction(
                        initial_position, 
                        imu_data[i, -1, 0:3],  # Last accelerometer input in the sequence
                        accel_bias, 
                        initial_velocity, 
                        gravity_vector.to(device), 
                        initial_quaternion, 
                        dt=dt, 
                        disturbance=disturbance
                    )
                    
                    # Compute the estimated quaternion using the corrected angular velocity
                    estimated_quaternion = compute_quaternion_with_correction(
                        initial_quaternion, 
                        imu_data[i, -1, 3:6],  # Last gyroscope input in the sequence
                        gyro_bias, 
                        dt=dt, 
                        disturbance=disturbance
                    )
                    
                    estimated_positions.append(estimated_position)
                    estimated_quaternions.append(estimated_quaternion)

                # Convert lists to tensors for further processing
                estimated_positions = torch.stack(estimated_positions)  # Convert list to tensor
                estimated_quaternions = torch.stack(estimated_quaternions)  # Convert list to tensor

                # Log raw outputs, denormalized outputs, estimated positions, and quaternions
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_index+1}/{len(train_dataloader)}]:")
                print(f"Model output: {predicted_increments[-1].cpu().detach().numpy()}")
                print(f"Estimated position: {estimated_positions[-1].cpu().detach().numpy()}")
                print(f"Ground truth position: {true_positions[-1].cpu().detach().numpy()}")
                print(f"Estimated quaternion: {estimated_quaternions[-1].cpu().detach().numpy()}")
                print(f"Ground truth quaternion: {true_quaternions[-1].cpu().detach().numpy()}")
                

            # Check if loss has grad_fn (it should!)
            assert loss.grad_fn is not None, "Loss does not have a grad_fn!"

            loss.backward()
            
            # Monitor gradients
            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        print(f"Gradients for {name}: {param.grad.norm()}")
            
            optimizer.step()
            epoch_loss += loss.item()
            
            data_counter += imu_data.size(0)
                

        avg_loss = epoch_loss / len(train_dataloader)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.14f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(val_dataloader, desc="Validating", leave=True)):
                batch_data = {k: v.to(device) for k, v in batch_data.items()}  # Move all data to GPU
                # Unpack the batch data
                imu_data = batch_data['imu_data']  # IMU data (input)
                biases = batch_data['biases']  # Biases (target)
                init_quaternions = batch_data['init_quaternion']  # Initial quaternions
                init_positions = batch_data['init_position']  # Initial positions
                init_velocities = batch_data['init_velocity']  # Initial velocities
                true_quaternions = batch_data['quaternion']  # True quaternions
                true_positions = batch_data['position']  # True positions
                true_velocities = batch_data['velocity']

                # Forward pass
                predicted_increments = model(imu_data, biases)
                predicted_biases = biases + predicted_increments

                # Calculate the loss
                loss = criterion(
                    true_quaternions=true_quaternions, 
                    true_positions=true_positions, 
                    predicted_biases=predicted_biases, 
                    batch_X=imu_data, 
                    init_positions=init_positions,  
                    init_quaternions=init_quaternions, 
                    init_velocities=init_velocities,  
                    gravity_vector=gravity_vector, 
                    sequence_length=seq_length
                )

                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation Loss: {avg_val_loss:.20f}')
        
        # Save Model Checkpoint with JIT Scripting
        model.to('cpu')
        scripted_model = torch.jit.script(model)
        model_path = os.path.join(save_dir, f"scripted_model_epoch_{epoch+1}.pth")
        scripted_model.save(model_path)
        print(f"Saved scripted model checkpoint to {model_path}")
        
        # Move model back to GPU if using CUDA
        if torch.cuda.is_available():
            model.to(device)
        
         # Save Losses to File
        with open(loss_log_file, "a") as f:
            f.write(f"Epoch {epoch+1}, Training Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}\n")
        #print(f"Saved losses to {loss_log_file}")

    print('----Finished Training----')

    # Save the trained model
    model.to('cpu')

    scripted_model = torch.jit.script(model)
    if dataset == 'euroc':
        model_path = "euroc_scripted_accel_model_10.pth"
    elif dataset == 'kitti':
        model_path = "kitti_scripted_accel_model_10.pth"
    #torch.save(model.state_dict(), model_path)
    scripted_model.save(model_path)

    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()