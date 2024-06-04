import numpy as np
import os
import pandas as pd
from scipy.linalg import logm

# Load your data
input_file = './ground_truth_estimate\data_aug_mh05.csv'
data = pd.read_csv(input_file)

print(data.columns)

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# Constants
g = np.array([0, 0, -9.81])  # gravitational acceleration in m/s^2

# Calculate Δt assuming constant sampling rate
timestamps = data['#timestamp']
Δt = 0.005  # Convert nanoseconds to seconds

# Storage for calculated biases
bias_gyro = []
bias_accel = []
bias_gyro_true = []
bias_accel_true = []

for i in range(len(data)-1):    
    
    if i % 100 == 0:
        print(i)
    
    # Save true values
    gyro_values = data.loc[i, [' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]']].values
    accel_values = data.loc[i, [' b_a_RS_S_x [m s^-2]', ' b_a_RS_S_y [m s^-2]', ' b_a_RS_S_z [m s^-2]']].values
    
    # Append NumPy arrays
    bias_gyro_true.append(gyro_values)
    bias_accel_true.append(accel_values)
    
    q1 = data.loc[i, [' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].values
    q2 = data.loc[i+1, [' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].values
    R1 = quaternion_to_rotation_matrix(q1)
    R2 = quaternion_to_rotation_matrix(q2)


    # Compute matrix logarithm of the rotation matrix difference
    R_diff = np.dot(R1.T, R2)
    log_R = logm(R_diff)
    omega_true = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]]) / Δt  # Extract angular velocities

    p1 = data.loc[i, [' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
    p2 = data.loc[i+1, [' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
    v1 = data.loc[i, [' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].values

    # Estimating biases using the provided formulas
    omega_measured = data.loc[i, ['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].values
    a_measured = data.loc[i, ['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].values

    b_gyro_est = omega_measured - omega_true
    bias_gyro.append(b_gyro_est)

    p_diff = p2 - p1
    v_avg = v1  # Assuming constant velocity over Δt
    accel_term = (R1.T) @ ((p_diff - v_avg * Δt - 0.5 * g * Δt**2) * (2 / Δt**2))
    #print(accel_term, a_measured)
    b_accel_est = a_measured - accel_term
    bias_accel.append(b_accel_est)

# Convert lists to numpy arrays
bias_gyro = np.array(bias_gyro)
bias_accel = np.array(bias_accel)
bias_gyro_true = np.array(bias_gyro_true)
bias_accel_true = np.array(bias_accel_true)

# Create DataFrames
bias_gyro_df = pd.DataFrame(bias_gyro, columns=['true_bias_gyro_x', 'true_bias_gyro_y', 'true_bias_gyro_z'])
bias_accel_df = pd.DataFrame(bias_accel, columns=['true_bias_accel_x', 'true_bias_accel_y', 'true_bias_accel_z'])
#bias_gyro_true_df = pd.DataFrame(bias_gyro_true, columns=['true_bias_gyro_x', 'true_bias_gyro_y', 'true_bias_gyro_z'])
#bias_accel_true_df = pd.DataFrame(bias_accel_true, columns=['true_bias_accel_x', 'true_bias_accel_y', 'true_bias_accel_z'])

# Concatenate DataFrames
biases_df = pd.concat([bias_gyro_df, bias_accel_df], axis=1)

# Extract base name of the input file
base_name = os.path.splitext(os.path.basename(input_file))[0]

# Create the output file name
output_file = f'computed_imu_biases_{base_name}.csv'

# Save to CSV
biases_df.to_csv(output_file, index=False)

# Optionally, output or save the results
print("\nEstimated Gyro Biases: \n", bias_gyro)
print("Estimated Accel Biases: \n", bias_accel)
print("\nTrue Gyro Biases: \n", bias_gyro_true)
print("True Accel Biases: \n", bias_accel_true)

print("\n===============================================\n")

print("Gyro bias mean estimated: \n",np.mean(bias_gyro[:, 0]))
print(np.mean(bias_gyro[:, 1]))
print(np.mean(bias_gyro[:, 2]))
print("\n Accel bias mean estimated: \n",np.mean(bias_accel[:, 0]))
print(np.mean(bias_accel[:, 1]))
print(np.mean(bias_accel[:, 2]))

print("\n===============================================\n")
print("Gyro bias mean true: \n",np.mean(bias_gyro_true[:, 0]))
print(np.mean(bias_gyro_true[:, 1]))
print(np.mean(bias_gyro_true[:, 2]))
print("\n Accel bias mean true: \n",np.mean(bias_accel_true[:, 0]))
print(np.mean(bias_accel_true[:, 1]))
print(np.mean(bias_accel_true[:, 2]))

