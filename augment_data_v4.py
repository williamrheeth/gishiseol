import pandas as pd
import numpy as np
import os

# Load the CSV files
################################################################################################
imu_file_path = './transformer_v4/raw_data/v203_train_odom-vins_estimator-imu_frame_data.csv'
df_aligned = pd.read_csv('./ground_truth_estimate/data_aug_v203.csv')
df_data = pd.read_csv('./transformer_v4/raw_data/v203data.csv')
################################################################################################

imu_file_name = os.path.basename(imu_file_path)
prefix = imu_file_name[:4]
df_imu_frame_data = pd.read_csv(imu_file_path)

# Extract necessary columns from imu_frame_data
df_odom = df_imu_frame_data[['.pose.pose.position.x', '.pose.pose.position.y', '.pose.pose.position.z',
                             '.pose.pose.orientation.w', '.pose.pose.orientation.x', '.pose.pose.orientation.y', '.pose.pose.orientation.z',
                             '.twist.twist.linear.x', '.twist.twist.linear.y', '.twist.twist.linear.z',
                             '.twist.twist.angular.x', '.twist.twist.angular.y', '.twist.twist.angular.z']]

# Extract only the first 6 elements from the covariance (accel bias: first 3, gyro bias: next 3)
df_biases = df_imu_frame_data['.pose.covariance'].apply(lambda x: eval(x))  # Convert the covariance string to list
df_biases_filtered = df_biases.apply(lambda cov: cov[:6])  # Select only the first 6 elements
df_biases_filtered = pd.DataFrame(df_biases_filtered.tolist(), columns=['b_a_RS_S_x [m s^-2]', 'b_a_RS_S_y [m s^-2]', 'b_a_RS_S_z [m s^-2]',
                                                                        'b_w_RS_S_x [rad s^-1]', 'b_w_RS_S_y [rad s^-1]', 'b_w_RS_S_z [rad s^-1]'])

# Extract acceleration from the first three values of twist.covariance
df_acceleration = df_imu_frame_data['.twist.covariance'].apply(lambda x: eval(x))  # Convert the covariance string to list
df_acceleration_filtered = df_acceleration.apply(lambda cov: cov[:3])  # Select only the first 3 elements (acceleration x, y, z)
df_acceleration_filtered = pd.DataFrame(df_acceleration_filtered.tolist(), columns=['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]'])

# Combine IMU data and biases
df_combined_odom = pd.concat([df_odom, df_acceleration_filtered, df_biases_filtered], axis=1)

# Rename columns to match the required format
df_combined_odom.columns = ['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []',
                            'v_RS_R_x [m s^-1]', 'v_RS_R_y [m s^-1]', 'v_RS_R_z [m s^-1]', 
                            'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
                            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]',
                            'b_a_RS_S_x [m s^-2]', 'b_a_RS_S_y [m s^-2]', 'b_a_RS_S_z [m s^-2]',
                            'b_w_RS_S_x [rad s^-1]', 'b_w_RS_S_y [rad s^-1]', 'b_w_RS_S_z [rad s^-1]'
                            ]

# Extract ground truth pose and orientation from data.csv with updated headings
df_truth = df_data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]', ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']]
df_truth.columns = ['true_p_x[m]', 'true_p_y[m]', 'true_p_z[m]', 'true_q_w[]', 'true_q_x[]', 'true_q_y[]', 'true_q_z[]']

# Set the tolerance for comparison based on observed precision
tolerance = 1e-9  # Adjust based on the decimal places observed in the aligned data

# Get the first values from the aligned dataset
first_aligned_p_x = df_aligned.iloc[0][' p_RS_R_x [m]']
first_aligned_p_y = df_aligned.iloc[0][' p_RS_R_y [m]']
first_aligned_p_z = df_aligned.iloc[0][' p_RS_R_z [m]']
first_aligned_w_x = df_aligned.iloc[0]['w_RS_S_x [rad s^-1]']
first_aligned_w_y = df_aligned.iloc[0]['w_RS_S_y [rad s^-1]']
first_aligned_w_z = df_aligned.iloc[0]['w_RS_S_z [rad s^-1]']

# Find the first index in the odom dataset where the position and gyro values match
odom_start_index = df_combined_odom[
    np.isclose(df_combined_odom['w_RS_S_x [rad s^-1]'], first_aligned_w_x, atol=tolerance) &
    np.isclose(df_combined_odom['w_RS_S_y [rad s^-1]'], first_aligned_w_y, atol=tolerance) &
    np.isclose(df_combined_odom['w_RS_S_z [rad s^-1]'], first_aligned_w_z, atol=tolerance)
].index[0]

# Find the first index in the truth dataset where the position matches the aligned data
truth_start_index = df_truth[
    np.isclose(df_truth['true_p_x[m]'], first_aligned_p_x, atol=tolerance) &
    np.isclose(df_truth['true_p_y[m]'], first_aligned_p_y, atol=tolerance) &
    np.isclose(df_truth['true_p_z[m]'], first_aligned_p_z, atol=tolerance)
].index[0]

print(first_aligned_w_x, first_aligned_w_y, first_aligned_w_z)
print(odom_start_index, truth_start_index)

# Slice the odom and truth datasets starting from the matching index
df_combined_odom_aligned = df_combined_odom.iloc[odom_start_index:].reset_index(drop=True)
df_combined_odom_aligned = df_combined_odom_aligned[df_combined_odom_aligned['w_RS_S_x [rad s^-1]'].shift() != df_combined_odom_aligned['w_RS_S_x [rad s^-1]']]
df_combined_odom_aligned = df_combined_odom_aligned.reset_index(drop=True)
df_truth_aligned = df_truth.iloc[truth_start_index:].reset_index(drop=True)

min_length = min(len(df_combined_odom_aligned), len(df_truth_aligned))

df_combined_odom_aligned = df_combined_odom_aligned.iloc[:min_length]
df_truth_aligned = df_truth_aligned.iloc[:min_length]

# Combine the aligned odom and truth datasets
df_final = pd.concat([df_combined_odom_aligned, df_truth_aligned], axis=1)

# Check where b_a_RS_S_x, b_a_RS_S_y, b_a_RS_S_z, b_w_RS_S_x, b_w_RS_S_y, b_w_RS_S_z are all zero
bias_zero_condition = (
    (df_final['b_a_RS_S_x [m s^-2]'] == 0) & (df_final['b_a_RS_S_y [m s^-2]'] == 0) & (df_final['b_a_RS_S_z [m s^-2]'] == 0) &
    (df_final['b_w_RS_S_x [rad s^-1]'] == 0) & (df_final['b_w_RS_S_y [rad s^-1]'] == 0) & (df_final['b_w_RS_S_z [rad s^-1]'] == 0)
)

# Find the first index where the biases are non-zero
first_non_zero_index = df_final[~bias_zero_condition].index[0]

# Crop the DataFrame from the first non-zero index onwards
df_final = df_final.iloc[first_non_zero_index:].reset_index(drop=True)

# Save the combined dataframe to a new CSV file
output_file_name = f'./transformer_v4/augmented_data/{prefix}.csv'
df_final.to_csv(output_file_name, index=False)

print(f"Data saved to {output_file_name}")
