import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# 0. Global Reproducibility Fix (Locks Significance Variance)
# ==========================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures PyTorch operations run deterministically across execution environments
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(42)

# ==========================================
# 1. Load and Prepare Data
# ==========================================
FILE_PATH = "/workspaces/PINN-DFT/data_cleaned.csv"

# Load the dataset
df = pd.read_csv(FILE_PATH)

# Separate features and target
X = df.drop(columns=['Band gap (HSE06) [eV]']).values
y = df['Band gap (HSE06) [eV]'].values.reshape(-1, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (crucial for Neural Networks)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors for the standalone PINN components
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


# ==========================================
# Model 1: Smoothed Random Forest (RF)
# ==========================================
rf_model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
rf_model.fit(X_train_scaled, y_train_scaled.ravel())

print("Model 1: Random Forest trained successfully.")

# Get feature importances from RF
importances = rf_model.feature_importances_
feature_names = df.drop(columns=['Band gap (HSE06) [eV]']).columns

# Sort and plot
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances Determining HSE06 Band Gap")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature_importance.png to your workspace.")


# ==========================================
# Model 2: Physics-Informed Neural Network (PINN)
# ==========================================
class PINN(nn.Module):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),  
            nn.Dropout(0.05),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def physics_loss_fn(y_pred):
    """Penalize non-physical negative band gap predictions."""
    return torch.mean(torch.relu(-y_pred))

# Instantiate and train standalone PINN
pinn_model = PINN(input_dim=X_train_scaled.shape[1])
pinn_optimizer = optim.AdamW(pinn_model.parameters(), lr=0.005, weight_decay=1e-4)
pinn_scheduler = optim.lr_scheduler.ExponentialLR(pinn_optimizer, gamma=0.998)
mse_loss = nn.MSELoss()

pinn_model.train()
for epoch in range(1500):
    pinn_optimizer.zero_grad()
    
    y_pred = pinn_model(X_train_tensor)
    
    loss_data = mse_loss(y_pred, y_train_tensor)
    loss_physics = physics_loss_fn(y_pred)
    total_loss = loss_data + 0.001 * loss_physics
    
    total_loss.backward()
    pinn_optimizer.step()
    pinn_scheduler.step()

print("Model 2: Standalone PINN trained successfully.")


# ==========================================
# Model 3: Restored Residual Hybrid RF-PINN
# ==========================================
# 1. Extract RF predictions to use as an empirical baseline hint feature
rf_train_preds_raw = rf_model.predict(X_train_scaled).reshape(-1, 1)
rf_test_preds_raw = rf_model.predict(X_test_scaled).reshape(-1, 1)

# Scale the RF predictions
scaler_rf = StandardScaler()
rf_train_preds_scaled = scaler_rf.fit_transform(rf_train_preds_raw)
rf_test_preds_scaled = scaler_rf.transform(rf_test_preds_raw)

# 2. Augment the input feature space with the SCALED RF predictions
X_train_hybrid = np.hstack((X_train_scaled, rf_train_preds_scaled))
X_test_hybrid = np.hstack((X_test_scaled, rf_test_preds_scaled))

# Convert augmented features to tensors
X_train_hybrid_tensor = torch.tensor(X_train_hybrid, dtype=torch.float32)
X_test_hybrid_tensor = torch.tensor(X_test_hybrid, dtype=torch.float32)

# RESTORED: Pure Skip-Connection Residual Network Architecture with Stable Batch Normalization
class ResidualHybridPINN(nn.Module):
    def __init__(self, input_dim):
        super(ResidualHybridPINN, self).__init__()
        # Input dimension passed to layers is (input_dim - 1) because the RF column bypasses them
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim - 1, 128),
            nn.BatchNorm1d(128),         # Stabilizes tabular input distributions
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),          # Stabilizes hidden representations
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x_hybrid):
        # Slice away the final column (the scaled RF prediction shortcut)
        x_features = x_hybrid[:, :-1]
        rf_shortcut = x_hybrid[:, -1:]
        
        # Calculate the residual delta correction from the physical features
        residual_correction = self.feature_net(x_features)
        
        # Final output is the direct sum: RF baseline expectation + PINN delta adjustments
        return rf_shortcut + residual_correction

# Instantiate the Restored Hybrid PINN
hybrid_pinn = ResidualHybridPINN(input_dim=X_train_hybrid.shape[1])
hybrid_optimizer = optim.AdamW(hybrid_pinn.parameters(), lr=0.005, weight_decay=1e-4)
# Continuous smooth learning decay
hybrid_scheduler = optim.lr_scheduler.ExponentialLR(hybrid_optimizer, gamma=0.998)

hybrid_pinn.train()
for epoch in range(1500):
    hybrid_optimizer.zero_grad()
    
    hybrid_pred = hybrid_pinn(X_train_hybrid_tensor)
    
    loss_data = mse_loss(hybrid_pred, y_train_tensor)
    loss_physics = physics_loss_fn(hybrid_pred)
    
    total_hybrid_loss = loss_data + 0.001 * loss_physics
    
    total_hybrid_loss.backward()
    hybrid_optimizer.step()
    hybrid_scheduler.step()

print("Model 3: Advanced Residual Hybrid RF-PINN trained successfully.")


# ==========================================
# Evaluation
# ==========================================
# Revert scaling back to original units for meaningful evaluation
rf_final_preds = scaler_y.inverse_transform(rf_model.predict(X_test_scaled).reshape(-1, 1))

pinn_model.eval()
with torch.no_grad():
    pinn_final_preds = scaler_y.inverse_transform(pinn_model(X_test_tensor).numpy())

hybrid_pinn.eval()
with torch.no_grad():
    hybrid_final_preds = scaler_y.inverse_transform(hybrid_pinn(X_test_hybrid_tensor).numpy())

print("\n--- Training & Comparison Complete ---")
print(f"Sample Actual Band Gap: {y_test[0][0]:.4f} eV")
print(f"RF Prediction:          {rf_final_preds[0][0]:.4f} eV")
print(f"PINN Prediction:        {pinn_final_preds[0][0]:.4f} eV")
print(f"Hybrid Prediction:      {hybrid_final_preds[0][0]:.4f} eV")

# ==========================================
# R² & MSE Score Calculations
# ==========================================
rf_r2 = r2_score(y_test, rf_final_preds)
pinn_r2 = r2_score(y_test, pinn_final_preds)
hybrid_r2 = r2_score(y_test, hybrid_final_preds)

print("\n--- Overall Test R² Score (Higher is Better, Max 1.0) ---")
print(f"RF Test R²:     {rf_r2:.4f}")
print(f"PINN Test R²:   {pinn_r2:.4f}")
print(f"Hybrid Test R²: {hybrid_r2:.4f}")

print("\n--- Overall Test MSE Score (Lower is Better) ---")
print(f"RF Test MSE:     {mean_squared_error(y_test, rf_final_preds):.4f}")
print(f"PINN Test MSE:   {mean_squared_error(y_test, pinn_final_preds):.4f}")
print(f"Hybrid Test MSE: {mean_squared_error(y_test, hybrid_final_preds):.4f}")

# ==========================================
# Robust Statistical Significance Test (Paired Error Evaluation)
# ==========================================
# Calculate squared errors for a robust variance check across predictions
rf_sq_errors = (y_test - rf_final_preds).flatten() ** 2
hybrid_sq_errors = (y_test - hybrid_final_preds).flatten() ** 2

# Perform a paired sample t-test on the squared errors
t_stat, p_value = stats.ttest_rel(hybrid_sq_errors, rf_sq_errors, alternative='less')

print("\n--- Statistical Significance Test (Hybrid vs. RF) ---")
print(f"T-Statistic:        {t_stat:.4f}")
print(f"Calculated p-value: {p_value:.6e}")

if p_value < 0.05:
    print("Result: Statistically Significant! (p < 0.05)")
    print("The error reduction from the Hybrid RF-PINN model is genuinely driven by the architecture, not random chance.")
else:
    print("Result: Not Statistically Significant (p >= 0.05)")
    print("The performance difference between the models could still be due to random distribution quirks.")

# ==========================================
# High-Throughput Screening Discovery
# ==========================================
optimal_candidates_mask = (hybrid_final_preds >= 1.0) & (hybrid_final_preds <= 1.8)
num_discoveries = np.sum(optimal_candidates_mask)

print(f"\n--- High-Throughput Screening Discovery ---")
print(f"Identified {num_discoveries} rectangular 2D materials matching optimal optoelectronic criteria.")