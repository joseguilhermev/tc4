import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM model definition
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # First LSTM layer
        self.bidirectional_lstm = nn.LSTM(
            input_size, hidden_size, num_layers=1, 
            batch_first=True, bidirectional=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        
        # Second LSTM layer
        self.lstm = nn.LSTM(
            hidden_size * 2, hidden_size, num_layers=1,
            batch_first=True, dropout=0.3
        )
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Dense layers with dropout
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.bn4 = nn.BatchNorm1d(32)
        
        # Output layer
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Bidirectional LSTM
        out, _ = self.bidirectional_lstm(x)
        out = self.bn1(out[:, -1, :])
        out = out.unsqueeze(1)  # Restore sequence dimension for next LSTM
        
        # Second LSTM layer
        out, _ = self.lstm(out)
        out = self.bn2(out[:, -1, :])
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.bn3(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.bn4(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

# Data loading and preprocessing
def load_data(symbol="AAPL", start_date="2000-01-01"):
    """
    Load stock data from Yahoo Finance starting from the given date.
    """
    df = yf.download(symbol, start=start_date, auto_adjust=True)
    df['Date'] = pd.to_datetime(df.index)
    df.sort_values('Date', inplace=True)
    
    # Calculate daily returns
    df['Return'] = df['Adj Close'].pct_change().fillna(0)
    
    return df

def preprocess_data(df, lookback=30):
    # Standardize the returns
    scaler = StandardScaler()
    df['Return_scaled'] = scaler.fit_transform(df[['Return']])
    
    # Create sliding windows for the input sequences
    X, y = [], []
    data = df['Return_scaled'].values
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y).reshape(-1, 1))
    
    return X, y, scaler

# Training and evaluation function
def train_and_evaluate(symbol="AAPL", start_date="2015-01-01", lookback=30, epochs=50, batch_size=32, 
                       train_split=0.7, val_split=0.15, test_split=0.15, hidden_size=128, learning_rate=0.001):
    # Load and preprocess data
    df = load_data(symbol, start_date)
    X, y, scaler = preprocess_data(df, lookback=lookback)
    
    # Split data into train, validation, and test sets
    total_samples = len(X)
    train_idx = int(train_split * total_samples)
    val_idx = train_idx + int(val_split * total_samples)
    
    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = StockLSTM(input_size=1, hidden_size=hidden_size, num_layers=2, output_size=1).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping with model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 10:  # patience: 10 epochs
                print("Early stopping triggered")
                break
    
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluation phase
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Convert predictions and targets back to original scale
    test_predictions = np.array(test_predictions).reshape(-1, 1)
    test_targets = np.array(test_targets).reshape(-1, 1)
    
    y_pred_test_inv = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(test_targets)
    
    # Calculate metrics
    mse = np.mean((y_pred_test_inv - y_test_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_test_inv - y_test_inv))
    
    print(f"Test Results - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Actual Returns", color="blue")
    plt.plot(y_pred_test_inv, label="Predicted Returns", color="red", linestyle="--")
    plt.title("Test Set: Actual vs Predicted Returns")
    plt.xlabel("Sample Index")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.show()
    
    # Save model
    os.makedirs(f"models/{symbol}", exist_ok=True)
    torch.save(model.state_dict(), f"models/{symbol}/model.pt")
    
    return {"mse": mse, "rmse": rmse, "mae": mae}

# Inference function
def predict_return(symbol="AAPL", lookback=30, hidden_size=128):
    model_path = f"models/{symbol}/model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load and preprocess data
    df = load_data(symbol)
    X, _, scaler = preprocess_data(df, lookback=lookback)
    
    # Get the last sequence for prediction
    last_sequence = X[-1].unsqueeze(0).to(device)
    
    # Load model and predict
    model = StockLSTM(input_size=1, hidden_size=hidden_size, num_layers=2, output_size=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        predicted_return_scaled = model(last_sequence).cpu().numpy()
    
    # Inverse transform
    predicted_return = scaler.inverse_transform(predicted_return_scaled)[0, 0]
    
    return predicted_return

# Example usage
if __name__ == "__main__":
    train_and_evaluate("AAPL", start_date="2015-01-01", epochs=50, batch_size=64)
