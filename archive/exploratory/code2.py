import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# 0. Global Reproducibility Fix 
# ==========================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(42)

# ==========================================
# 1. Load Data
# ==========================================
FILE_PATH = "/workspaces/PINN-DFT/data_cleaned.csv"
df = pd.read_csv(FILE_PATH)

X = df.drop(columns=['Band gap (HSE06) [eV]']).values
y = df['Band gap (HSE06) [eV]'].values.reshape(-1, 1)

# ==========================================
# Model Architectures
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

class ResidualHybridPINN(nn.Module):
    def __init__(self, input_dim):
        super(ResidualHybridPINN, self).__init__()
        # input_dim here includes raw features, rf shortcut, AND the new interaction features
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim - 1, 128),
            nn.BatchNorm1d(128),         
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),          
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x_hybrid):
        # Slice away only the single RF shortcut column at the very end
        x_features = x_hybrid[:, :-1]
        rf_shortcut = x_hybrid[:, -1:]
        
        residual_correction = self.feature_net(x_features)
        return rf_shortcut + residual_correction

def physics_loss_fn(y_pred):
    """Penalize non-physical negative band gap predictions."""
    return torch.mean(torch.relu(-y_pred))

# ==========================================
# 2. 5-Fold Cross-Validation Loop
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_rf_r2, fold_pinn_r2, fold_hybrid_r2 = [], [], []
fold_rf_mse, fold_pinn_mse, fold_hybrid_mse = [], [], []

all_rf_sq_errors = []
all_hybrid_sq_errors = []

print("Starting 5-Fold Statistical Validation with Empirical Interaction Tuning...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n--- Training Fold {fold + 1}/5 ---")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # --- Train Model 1: Random Forest ---
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train_scaled.ravel())
    
    # --- Train Model 2: Standalone PINN ---
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
        total_pinn_loss = loss_data + 0.001 * loss_physics
        total_pinn_loss.backward()
        pinn_optimizer.step()
        pinn_scheduler.step()
        
    # --- Train Model 3: Advanced Residual Hybrid RF-PINN ---
    rf_train_preds_raw = rf_model.predict(X_train_scaled).reshape(-1, 1)
    rf_test_preds_raw = rf_model.predict(X_test_scaled).reshape(-1, 1)
    
    scaler_rf = StandardScaler()
    rf_train_preds_scaled = scaler_rf.fit_transform(rf_train_preds_raw)
    rf_test_preds_scaled = scaler_rf.transform(rf_test_preds_raw)
    
    # ACCURACY BOOST addition: Generate cross-over non-linear empirical features
    # Multiplies the scaled RF baseline against all scaled raw physical inputs
    rf_train_interactions = X_train_scaled * rf_train_preds_scaled
    rf_test_interactions = X_test_scaled * rf_test_preds_scaled
    
    # Stack: [Raw Features, Interaction Features, Single RF Shortcut]
    X_train_hybrid = np.hstack((X_train_scaled, rf_train_interactions, rf_train_preds_scaled))
    X_test_hybrid = np.hstack((X_test_scaled, rf_test_interactions, rf_test_preds_scaled))
    
    X_train_hybrid_tensor = torch.tensor(X_train_hybrid, dtype=torch.float32)
    X_test_hybrid_tensor = torch.tensor(X_test_hybrid, dtype=torch.float32)
    
    hybrid_pinn = ResidualHybridPINN(input_dim=X_train_hybrid.shape[1])
    hybrid_optimizer = optim.AdamW(hybrid_pinn.parameters(), lr=0.005, weight_decay=1e-4)
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
        
    # --- Evaluation ---
    rf_final_preds = scaler_y.inverse_transform(rf_model.predict(X_test_scaled).reshape(-1, 1))
    
    pinn_model.eval()
    with torch.no_grad():
        pinn_final_preds = scaler_y.inverse_transform(pinn_model(X_test_tensor).numpy())
        
    hybrid_pinn.eval()
    with torch.no_grad():
        hybrid_final_preds = scaler_y.inverse_transform(hybrid_pinn(X_test_hybrid_tensor).numpy())
        
    fold_rf_r2.append(r2_score(y_test, rf_final_preds))
    fold_pinn_r2.append(r2_score(y_test, pinn_final_preds))
    fold_hybrid_r2.append(r2_score(y_test, hybrid_final_preds))
    
    fold_rf_mse.append(mean_squared_error(y_test, rf_final_preds))
    fold_pinn_mse.append(mean_squared_error(y_test, pinn_final_preds))
    fold_hybrid_mse.append(mean_squared_error(y_test, hybrid_final_preds))
    
    all_rf_sq_errors.extend(((y_test - rf_final_preds).flatten() ** 2).tolist())
    all_hybrid_sq_errors.extend(((y_test - hybrid_final_preds).flatten() ** 2).tolist())

# ==========================================
# 3. Comprehensive Statistical Reporting
# ==========================================
print("\n==================================================")
print("         CROSS-VALIDATION BENCHMARK REPORT        ")
print("==================================================")
print(f"Mean Random Forest R²:      {np.mean(fold_rf_r2):.4f} ± {np.std(fold_rf_r2):.4f}")
print(f"Mean Standalone PINN R²:    {np.mean(fold_pinn_r2):.4f} ± {np.std(fold_pinn_r2):.4f}")
print(f"Mean Residual Hybrid R²:    {np.mean(fold_hybrid_r2):.4f} ± {np.std(fold_hybrid_r2):.4f}")
print("--------------------------------------------------")
print(f"Mean Random Forest MSE:     {np.mean(fold_rf_mse):.4f}")
print(f"Mean Standalone PINN MSE:   {np.mean(fold_pinn_mse):.4f}")
print(f"Mean Residual Hybrid MSE:   {np.mean(fold_hybrid_mse):.4f}")

# Perform the unified paired test over the complete distribution profiles
t_stat, p_value = stats.ttest_rel(all_hybrid_sq_errors, all_rf_sq_errors, alternative='less')

print("\n--- Unified Cross-Validated Significance Test (Hybrid vs. RF) ---")
print(f"Aggregated T-Statistic: {t_stat:.4f}")
print(f"Calculated p-value:     {p_value:.6e}")

if p_value < 0.05:
    print("Result: Statistically Significant! (p < 0.05)")
    print("The error reduction from your Hybrid RF-PINN architecture holds systemic validity across the entire dataset distribution.")
else:
    print("Result: Not Statistically Significant (p >= 0.05)")
    print("The metrics could still be influenced by localized structural anomalies.")



# ==========================================
# 4. Generate Feature Importance Checkpoint
# ==========================================

"""
importances = rf_model.feature_importances_
feature_names = df.drop(columns=['Band gap (HSE06) [eV]']).columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances Determining HSE06 Band Gap")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\nSaved cross-validated feature_importance.png to workspace.")

"""
