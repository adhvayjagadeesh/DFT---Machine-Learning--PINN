import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
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

class PureDeepMLP(nn.Module):
    def __init__(self, input_dim):
        super(PureDeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class PyramidHybridPINN(nn.Module):
    """
    UPGRADE 1: Multi-Scale Residual Feature Pyramid Network
    Slices input categories to feed specific layers directly rather than 
    forcing a flat representation layer to parse complex tree outputs from scratch.
    """
    def __init__(self, raw_dim):
        super(PyramidHybridPINN, self).__init__()
        # Layer 1 processes raw features + interaction features
        self.input_layer = nn.Sequential(
            nn.Linear(raw_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        # Layer 2 aggregates the processed representation
        self.hidden_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
        self.final_projection = nn.Linear(64, 1)

    def forward(self, x_hybrid):
        # Slice vectors dynamically based on setup boundaries
        # x_hybrid structure: [raw_features (raw_dim), interactions (raw_dim), tree_shortcut (1)]
        raw_dim = (x_hybrid.shape[1] - 1) // 2
        
        x_base_features = x_hybrid[:, :raw_dim * 2]
        tree_shortcut = x_hybrid[:, -1:]
        
        # Build residual hierarchy
        x_representation = self.input_layer(x_base_features)
        x_compressed = self.hidden_layer(x_representation)
        residual_correction = self.final_projection(x_compressed)
        
        return tree_shortcut + residual_correction

def advanced_gated_physics_loss(y_pred):
    """
    UPGRADE 2: Boundary-Aware Gated Physics Loss
    Applies non-linear exponential grading near physical boundaries 
    to preserve fine structural accuracy without muddying global gradients.
    """
    base_violation = torch.relu(-y_pred)
    # Exponential scaling activation if predictions get non-physically close to zero
    boundary_buffer = torch.exp(torch.relu(0.05 - y_pred)) - 1.0
    return torch.mean(base_violation + 0.1 * boundary_buffer)

# ==========================================
# 2. 5-Fold Cross-Validation Loop
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics = {
    'rf':          {'r2': [], 'mse': []},
    'gbr':         {'r2': [], 'mse': []},
    'svr':         {'r2': [], 'mse': []},
    'mlp':         {'r2': [], 'mse': []},
    'pinn':        {'r2': [], 'mse': []},
    'hybrid_rf':   {'r2': [], 'mse': []},
    'hybrid_gbr':  {'r2': [], 'mse': []}
}

all_gbr_sq_errors = []
all_hybrid_gbr_sq_errors = []

print("Starting Advanced Pyramid & Gated Physics Benchmarking Suite...")

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
    mse_loss = nn.MSELoss()
    
    # 1. Random Forest Base
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train_scaled.ravel())
    
    # 2. Gradient Boosting Base
    gbr_model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
    gbr_model.fit(X_train_scaled, y_train_scaled.ravel())
    
    # 3. Support Vector Regression
    svr_model = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    svr_model.fit(X_train_scaled, y_train_scaled.ravel())
    
    # 4. Pure Deep MLP
    mlp_model = PureDeepMLP(input_dim=X_train_scaled.shape[1])
    mlp_optimizer = optim.AdamW(mlp_model.parameters(), lr=0.005, weight_decay=1e-4)
    mlp_model.train()
    for epoch in range(1200):
        mlp_optimizer.zero_grad()
        loss = mse_loss(mlp_model(X_train_tensor), y_train_tensor)
        loss.backward()
        mlp_optimizer.step()
        
    # 5. Standalone PINN
    pinn_model = PINN(input_dim=X_train_scaled.shape[1])
    pinn_optimizer = optim.AdamW(pinn_model.parameters(), lr=0.005, weight_decay=1e-4)
    pinn_scheduler = optim.lr_scheduler.ExponentialLR(pinn_optimizer, gamma=0.998)
    pinn_model.train()
    for epoch in range(1500):
        pinn_optimizer.zero_grad()
        y_pred = pinn_model(X_train_tensor)
        total_pinn_loss = mse_loss(y_pred, y_train_tensor) + 0.001 * advanced_gated_physics_loss(y_pred)
        total_pinn_loss.backward()
        pinn_optimizer.step()
        pinn_scheduler.step()
        
    # 6. Hybrid Model A: Optimized RF-PINN (Pyramid Variant)
    rf_train_preds = rf_model.predict(X_train_scaled).reshape(-1, 1)
    rf_test_preds = rf_model.predict(X_test_scaled).reshape(-1, 1)
    scaler_rf = StandardScaler()
    rf_train_scaled_p = scaler_rf.fit_transform(rf_train_preds)
    rf_test_scaled_p = scaler_rf.transform(rf_test_preds)
    
    X_train_hybrid_rf = np.hstack((X_train_scaled, X_train_scaled * rf_train_scaled_p, rf_train_scaled_p))
    X_test_hybrid_rf = np.hstack((X_test_scaled, X_test_scaled * rf_test_scaled_p, rf_test_scaled_p))
    
    hybrid_rf = PyramidHybridPINN(raw_dim=X_train_scaled.shape[1])
    optimizer_rf = optim.AdamW(hybrid_rf.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler_rf = optim.lr_scheduler.ExponentialLR(optimizer_rf, gamma=0.998)
    
    hybrid_rf.train()
    for epoch in range(1500):
        optimizer_rf.zero_grad()
        pred = hybrid_rf(torch.tensor(X_train_hybrid_rf, dtype=torch.float32))
        loss_data = mse_loss(pred, y_train_tensor)
        with torch.no_grad(): adaptive_w = 0.001 * (loss_data.item() + 1e-5)
        total_loss = loss_data + adaptive_w * advanced_gated_physics_loss(pred)
        total_loss.backward()
        optimizer_rf.step()
        scheduler_rf.step()

    # 7. Hybrid Model B: Upgraded GBR-PINN (Pyramid Variant)
    gbr_train_preds = gbr_model.predict(X_train_scaled).reshape(-1, 1)
    gbr_test_preds = gbr_model.predict(X_test_scaled).reshape(-1, 1)
    scaler_gbr = StandardScaler()
    gbr_train_scaled_p = scaler_gbr.fit_transform(gbr_train_preds)
    gbr_test_scaled_p = scaler_gbr.transform(gbr_test_preds)
    
    X_train_hybrid_gbr = np.hstack((X_train_scaled, X_train_scaled * gbr_train_scaled_p, gbr_train_scaled_p))
    X_test_hybrid_gbr = np.hstack((X_test_scaled, X_test_scaled * gbr_test_scaled_p, gbr_test_scaled_p))
    
    hybrid_gbr = PyramidHybridPINN(raw_dim=X_train_scaled.shape[1])
    optimizer_gbr = optim.AdamW(hybrid_gbr.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler_gbr = optim.lr_scheduler.ExponentialLR(optimizer_gbr, gamma=0.998)
    
    hybrid_gbr.train()
    for epoch in range(1500):
        optimizer_gbr.zero_grad()
        pred = hybrid_gbr(torch.tensor(X_train_hybrid_gbr, dtype=torch.float32))
        loss_data = mse_loss(pred, y_train_tensor)
        with torch.no_grad(): adaptive_w = 0.001 * (loss_data.item() + 1e-5)
        total_loss = loss_data + adaptive_w * advanced_gated_physics_loss(pred)
        total_loss.backward()
        optimizer_gbr.step()
        scheduler_gbr.step()

    # --- Evaluation ---
    preds = {
        'rf':  scaler_y.inverse_transform(rf_model.predict(X_test_scaled).reshape(-1, 1)),
        'gbr': scaler_y.inverse_transform(gbr_model.predict(X_test_scaled).reshape(-1, 1)),
        'svr': scaler_y.inverse_transform(svr_model.predict(X_test_scaled).reshape(-1, 1))
    }
    
    mlp_model.eval()
    pinn_model.eval()
    hybrid_rf.eval()
    hybrid_gbr.eval()
    with torch.no_grad():
        preds['mlp'] = scaler_y.inverse_transform(mlp_model(X_test_tensor).numpy())
        preds['pinn'] = scaler_y.inverse_transform(pinn_model(X_test_tensor).numpy())
        preds['hybrid_rf'] = scaler_y.inverse_transform(hybrid_rf(torch.tensor(X_test_hybrid_rf, dtype=torch.float32)).numpy())
        preds['hybrid_gbr'] = scaler_y.inverse_transform(hybrid_gbr(torch.tensor(X_test_hybrid_gbr, dtype=torch.float32)).numpy())
        
    for model_name in metrics.keys():
        metrics[model_name]['r2'].append(r2_score(y_test, preds[model_name]))
        metrics[model_name]['mse'].append(mean_squared_error(y_test, preds[model_name]))
        
    all_gbr_sq_errors.extend(((y_test - preds['gbr']).flatten() ** 2).tolist())
    all_hybrid_gbr_sq_errors.extend(((y_test - preds['hybrid_gbr']).flatten() ** 2).tolist())

# ==========================================
# 3. Comprehensive Performance Evaluation
# ==========================================
print("\n==================================================================")
print("             GLOBAL 5-FOLD BENCHMARK HARNESS REPORT               ")
print("==================================================================")
for model_name in metrics.keys():
    mean_r2 = np.mean(metrics[model_name]['r2'])
    std_r2 = np.std(metrics[model_name]['r2'])
    mean_mse = np.mean(metrics[model_name]['mse'])
    print(f"Model: {model_name.upper():<10} | R²: {mean_r2:.4f} ± {std_r2:.4f} | MSE: {mean_mse:.4f}")
print("==================================================================")

# Paired test checking if Hybrid GBR beats pure GBR
t_stat, p_value = stats.ttest_rel(all_hybrid_gbr_sq_errors, all_gbr_sq_errors, alternative='less')
print("\n--- Significance Verification (Hybrid GBR vs. Standalone GBR) ---")
print(f"Aggregated T-Statistic: {t_stat:.4f} | Calculated p-value: {p_value:.6e}")