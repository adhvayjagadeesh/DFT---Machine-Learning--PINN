import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. Load and Prepare Data
# ==========================================
FILE_PATH = "/workspaces/PINN-DFT/data_cleaned.csv"

df = pd.read_csv(FILE_PATH)

X = df.drop(columns=['Band gap (HSE06) [eV]']).values
y = df['Band gap (HSE06) [eV]'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

print(f"Data prepared successfully. Evaluating baselines on {X_test.shape[0]} test samples...\n")

# ==========================================
# Baseline 1: Naive Mean Baseline
# ==========================================
# Simply predicts the average value of the training target every single time
mean_value = np.mean(y_train)
baseline_mean_preds = np.full_like(y_test, mean_value)

# ==========================================
# Baseline 2: Standard Linear Regression
# ==========================================
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled.ravel())
baseline_lr_preds = scaler_y.inverse_transform(lr_model.predict(X_test_scaled).reshape(-1, 1))

# ==========================================
# Baseline 3: Standard Vanilla Neural Network (No Physics, No RF)
# ==========================================
class VanillaMLP(nn.Module):
    def __init__(self, input_dim):
        super(VanillaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

vanilla_model = VanillaMLP(input_dim=X_train_scaled.shape[1])
optimizer = optim.AdamW(vanilla_model.parameters(), lr=0.005, weight_decay=1e-4)
mse_loss = nn.MSELoss()

vanilla_model.train()
for epoch in range(1500):
    optimizer.zero_grad()
    y_pred = vanilla_model(X_train_tensor)
    loss = mse_loss(y_pred, y_train_tensor) # Pure data loss, completely ignores physics constraints
    loss.backward()
    optimizer.step()

vanilla_model.eval()
with torch.no_grad():
    baseline_nn_preds = scaler_y.inverse_transform(vanilla_model(X_test_tensor).numpy())

# ==========================================
# Compile and Print Baseline Results
# ==========================================
print("==========================================================")
print("                  BASELINE MODEL RESULTS                  ")
print("==========================================================")

# R² Scores
r2_mean = r2_score(y_test, baseline_mean_preds)
r2_lr = r2_score(y_test, baseline_lr_preds)
r2_nn = r2_score(y_test, baseline_nn_preds)

print(f"1. Naive Mean Predictor R²:      {r2_mean:.4f} (Expect ~0.0000)")
print(f"2. Linear Regression R²:         {r2_lr:.4f}")
print(f"3. Vanilla Neural Network R²:    {r2_nn:.4f}")

# MSE Scores
mse_mean = mean_squared_error(y_test, baseline_mean_preds)
mse_lr = mean_squared_error(y_test, baseline_lr_preds)
mse_nn = mean_squared_error(y_test, baseline_nn_preds)

print("\n----------------------------------------------------------")
print(f"1. Naive Mean Predictor MSE:     {mse_mean:.4f}")
print(f"2. Linear Regression MSE:        {mse_lr:.4f}")
print(f"3. Vanilla Neural Network MSE:   {mse_nn:.4f}")
print("==========================================================")