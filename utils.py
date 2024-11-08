import torch

def quaternion_product(q, r):
    """
    Compute the product of two quaternions.
    q and r can be single quaternions of shape (4,)
    or batched quaternions of shape (batch_size, 4).
    """
    if q.ndim == 2 and r.ndim == 2:  # Handle batched quaternions
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        r0, r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    else:  # Handle single quaternion case
        q0, q1, q2, q3 = q
        r0, r1, r2, r3 = r

    # Calculate the quaternion product
    t0 = q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3
    t1 = q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2
    t2 = q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1
    t3 = q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0

    if q.ndim == 2 and r.ndim == 2:  # Stack for batched case
        return torch.stack([t0, t1, t2, t3], dim=1)  # Shape: (batch_size, 4)
    else:
        return torch.stack([t0, t1, t2, t3], dim=0)  # Shape: (4,)

def quaternion_conjugate(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quaternion_log(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, v = q[..., 0], q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True)
    v_norm = torch.where(v_norm > 1e-6, v_norm, torch.ones_like(v_norm) * 1e-6)
    angle = 2.0 * torch.atan2(v_norm, w)
    return angle[..., None] * (v / v_norm)

def denormalize_model_output_iqr(scaled_output, bias_median, bias_iqr):
    return scaled_output * bias_iqr + bias_median

def quaternion_rotate(q, v):
    """
    Rotate vector `v` by quaternion `q`.
    q should be a quaternion of shape (4,) or batched (batch_size, 4)
    v should be a vector of shape (3,) or batched (batch_size, 3)
    """
    if q.ndim == 2 and v.ndim == 2:  # Handle batched data
        # Convert vector v into a quaternion with 0 as the scalar part
        v_quat = torch.cat([torch.zeros(v.shape[0], 1, device=v.device), v], dim=1)  # Shape: (batch_size, 4)
        q_conj = quaternion_conjugate(q)  # Conjugate of quaternion q
    else:  # Handle single quaternion case
        v_quat = torch.cat([torch.tensor([0.0], device=v.device), v], dim=0)  # Shape: (4,)
        q_conj = quaternion_conjugate(q)

    # Apply the rotation: q * v_quat * q_conj
    rotated_v = quaternion_product(quaternion_product(q, v_quat), q_conj)

    return rotated_v[:, 1:] if rotated_v.ndim == 2 else rotated_v[1:]  # Return only the vector part

def compute_quaternion_with_correction(current_quaternion, angular_velocity, bias, dt=0.005, disturbance=0.0):
    angular_velocity_corrected = angular_velocity - bias - disturbance
    theta = torch.norm(angular_velocity_corrected, p=2, dim=-1, keepdim=True) * dt
    half_theta = theta / 2
    w = torch.cos(half_theta)
    norm_corrected = torch.norm(angular_velocity_corrected, p=2, dim=-1, keepdim=True)
    norm_corrected = torch.clamp(norm_corrected, min=1e-6)  # Avoid zero division
    xyz = torch.sin(half_theta) * angular_velocity_corrected / norm_corrected
    rotation_correction = torch.cat((w, xyz), dim=-1)

    rotation_correction = rotation_correction / torch.norm(rotation_correction, p=2, dim=-1, keepdim=True)
    current_quaternion = current_quaternion / torch.norm(current_quaternion, p=2, dim=-1, keepdim=True)

    w1, x1, y1, z1 = current_quaternion.unbind(dim=-1)
    w2, x2, y2, z2 = rotation_correction.unbind(dim=-1)

    new_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    new_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    new_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    new_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    updated_rotation = torch.stack((new_w, new_x, new_y, new_z), dim=-1).float()
    return updated_rotation


def compute_acceleration_with_correction(current_position, acceleration, bias, velocity, gravity, quaternion, dt=0.005, disturbance=0.0):
    acceleration_corrected = acceleration - bias - disturbance
    rotated_term = 0.5 * quaternion_rotate(quaternion, acceleration_corrected) * (dt**2)

    return current_position + velocity * dt + 0.5 * gravity * (dt ** 2) + rotated_term






###########################################################################################################

def normalize(data):
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True)
    return (data - mean) / std, mean, std

def denormalize(data, mean, std):
    return data * std + mean