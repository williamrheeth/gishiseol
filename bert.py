import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
data_path = 'ground_truth_estimate/data_aug.csv'
data = pd.read_csv(data_path)

# Prepare sequences
sequences = []
sequence_length = 10  # Adjusted sequence length based on typical time-series data

for i in range(len(data) - sequence_length):
    sequence = data.iloc[i:i+sequence_length][[
        'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]', 
        'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 
        ' b_w_RS_S_x [rad s^-1]', ' b_w_RS_S_y [rad s^-1]', ' b_w_RS_S_z [rad s^-1]'
    ]].values
    sequences.append(sequence)

sequences = np.array(sequences)
X = sequences[:, :, :-3]  # acceleration, angular velocity, previous gyro bias
y = sequences[:, -1, -3:]  # gyro bias

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Split data
X_test = X[:100]
y_test = y[:100]
X_train = X[100:]
y_train = y[100:]

# Define dataset class
class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = IMUDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = IMUDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define BERT model for IMU data
class LIMUBertModel4Pretrain(nn.Module):
    def __init__(self, hyper_params):
        super(LIMUBertModel4Pretrain, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hyper_params['d_model'], 
            nhead=hyper_params['nhead']
        )
        self.decoder = nn.Linear(hyper_params['d_model'], 6)
    
    def forward(self, x, mask_pos=None):
        x = self.encoder(x)
        if mask_pos is not None:
            x = x[:, mask_pos, :]
        x = self.decoder(x)
        return x

class Model(pl.LightningModule):
    def __init__(self, config):
        super(Model, self).__init__()
        self.save_hyperparameters("config")
        self.starting_learning_rate = float(config['model']['hyper_params']['starting_learning_rate'])
        self.hyper_params = config['model']['hyper_params']
        self.limu_bert_mlm = LIMUBertModel4Pretrain(self.hyper_params)
        self.limu_bert_nsp = LIMUBertModel4Pretrain(self.hyper_params)
        self.mse_loss = F.mse_loss
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        X, y = batch
        mask_seqs, masked_pos, gt_masked_seq, normed_input_imu, normed_future_imu = self._prepare_data(X)

        # MLM task
        hat_imu_MLM = self.limu_bert_mlm(mask_seqs, masked_pos)
        MLM_loss = self.mse_loss(gt_masked_seq, hat_imu_MLM) * float(self.hyper_params['mlm_loss_weights'])

        # Denoise task
        hat_imu_denoise = self.limu_bert_mlm(normed_input_imu)
        denoise_loss = self.mse_loss(normed_input_imu[:, -1, :], hat_imu_denoise[:, -1, :]) * float(self.hyper_params['denoise_loss_weights'])

        # NSP task
        hat_imu_future = self.limu_bert_nsp(normed_input_imu)
        hat_imu_future_denoised = self.limu_bert_nsp(hat_imu_denoise)
        NSP_loss = (self.mse_loss(normed_future_imu, hat_imu_future) + self.mse_loss(hat_imu_future_denoised, hat_imu_future)) * float(self.hyper_params['nsp_loss_weights'])

        loss = MLM_loss + denoise_loss + NSP_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MLM_loss", MLM_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_denoise_loss", denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_NSP_loss", NSP_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        mask_seqs, masked_pos, gt_masked_seq, normed_input_imu, normed_future_imu = self._prepare_data(X)

        # MLM task
        hat_imu_MLM = self.limu_bert_mlm(mask_seqs, masked_pos)
        MLM_loss = self.mse_loss(gt_masked_seq, hat_imu_MLM) * float(self.hyper_params['mlm_loss_weights'])

        # Denoise task
        hat_imu_denoise = self.limu_bert_mlm(normed_input_imu)
        denoise_loss = self.mse_loss(normed_input_imu[:, -1, :], hat_imu_denoise[:, -1, :]) * float(self.hyper_params['denoise_loss_weights'])

        # NSP task
        hat_imu_future = self.limu_bert_nsp(normed_input_imu)
        hat_imu_future_denoised = self.limu_bert_nsp(hat_imu_denoise)
        NSP_loss = (self.mse_loss(normed_future_imu, hat_imu_future) + self.mse_loss(hat_imu_future_denoised, hat_imu_future)) * float(self.hyper_params['nsp_loss_weights'])

        loss = MLM_loss + denoise_loss + NSP_loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_MLM_loss", MLM_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_denoise_loss", denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_NSP_loss", NSP_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.starting_learning_rate)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(self.hyper_params['T_0']), T_mult=int(self.hyper_params['T_mult']), eta_min=float(self.hyper_params['eta_min'])),
            "interval": "epoch",
            "frequency": 1,
            'name': 'learning_rate'
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _prepare_data(self, X):
        # Dummy data preparation function, replace with actual logic
        mask_seqs = X.clone()
        masked_pos = torch.randint(0, X.size(1), (X.size(0), int(X.size(1) * 0.15)))
        gt_masked_seq = X.clone()
        normed_input_imu = X.clone()
        normed_future_imu = X.clone()
        return mask_seqs, masked_pos, gt_masked_seq, normed_input_imu, normed_future_imu

# Configuration
config = {
    "model": {
        "hyper_params": {
            "starting_learning_rate": 0.001,
            "d_model": 64,
            "nhead": 8,
            "mlm_loss_weights": 1.0,
            "denoise_loss_weights": 1.0,
            "nsp_loss_weights": 1.0,
            "T_0": 10,
            "T_mult": 2,
            "eta_min": 1e-6
        }
    }
}

# Initialize model
model = Model(config)

# Training
trainer = pl.Trainer(max_epochs=10, devices=1, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
trainer.fit(model, train_loader, val_dataloaders=test_loader)

# Testing
trainer.test(model, test_loader)