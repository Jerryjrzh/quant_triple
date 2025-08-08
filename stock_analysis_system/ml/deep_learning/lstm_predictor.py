"""
LSTM Stock Price Predictor

Advanced LSTM model for stock price time series prediction with:
- Multi-layer LSTM architecture
- Attention mechanisms
- Dropout regularization
- Batch normalization
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """LSTM model configuration"""
    input_size: int = 5  # OHLCV
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    sequence_length: int = 60
    prediction_horizon: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    use_attention: bool = True
    use_batch_norm: bool = True


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        # attended_output: (batch_size, hidden_size)
        
        return attended_output, attention_weights


class LSTMStockModel(nn.Module):
    """Advanced LSTM model for stock prediction"""
    
    def __init__(self, config: LSTMConfig):
        super(LSTMStockModel, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Batch normalization
        if config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        
        # Attention mechanism
        if config.use_attention:
            self.attention = AttentionLayer(config.hidden_size)
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc2 = nn.Linear(config.hidden_size // 2, config.prediction_horizon)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size)
        
        if self.config.use_attention:
            # Use attention mechanism
            attended_out, attention_weights = self.attention(lstm_out)
            output = attended_out
        else:
            # Use last hidden state
            output = lstm_out[:, -1, :]
        
        # Batch normalization
        if self.config.use_batch_norm:
            output = self.batch_norm(output)
        
        # Fully connected layers
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


class LSTMStockPredictor:
    """LSTM-based stock price predictor"""
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
        
        logger.info(f"Initialized LSTM predictor with device: {self.device}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Expected columns: ['open', 'high', 'low', 'close', 'volume']
        features = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in features):
            raise ValueError(f"Data must contain columns: {features}")
        
        # Sort by date
        data = data.sort_index()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(data[features].values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Input sequence
            X.append(scaled_data[i:(i + self.config.sequence_length)])
            
            # Target: next prediction_horizon close prices
            target_start = i + self.config.sequence_length
            target_end = target_start + self.config.prediction_horizon
            y.append(scaled_data[target_start:target_end, 3])  # Close price index
        
        return np.array(X), np.array(y)
    
    def create_data_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch data loader"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the LSTM model"""
        logger.info("Starting LSTM model training...")
        
        # Prepare training data
        X_train, y_train = self.prepare_data(train_data)
        train_loader = self.create_data_loader(X_train, y_train, shuffle=True)
        
        # Prepare validation data
        val_loader = None
        if val_data is not None:
            X_val, y_val = self.prepare_data(val_data)
            val_loader = self.create_data_loader(X_val, y_val, shuffle=False)
        
        # Initialize model
        self.model = LSTMStockModel(self.config).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params({
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'sequence_length': self.config.sequence_length,
                'prediction_horizon': self.config.prediction_horizon,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'use_attention': self.config.use_attention
            })
            
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            outputs = self.model(batch_X)
                            val_loss += criterion(outputs, batch_y).item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    scheduler.step(avg_val_loss)
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(self.model.state_dict(), 'best_lstm_model.pth')
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if val_loader else 0,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                # Store training history
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if val_loader else None,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Load best model
            if val_loader is not None:
                self.model.load_state_dict(torch.load('best_lstm_model.pth'))
            
            # Log model
            mlflow.pytorch.log_model(self.model, "lstm_model")
        
        logger.info("LSTM model training completed")
        
        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss if val_loader else None,
            'best_val_loss': best_val_loss if val_loader else None,
            'epochs_trained': len(self.training_history)
        }
    
    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> Dict[str, Any]:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Prepare data
        features = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = self.scaler.transform(data[features].values)
        
        predictions = []
        confidence_intervals = []
        
        # Use the last sequence_length data points for prediction
        current_sequence = scaled_data[-self.config.sequence_length:]
        
        with torch.no_grad():
            for step in range(steps_ahead):
                # Prepare input
                X = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Make prediction
                pred = self.model(X)
                pred_np = pred.cpu().numpy()[0]
                
                # Store prediction
                predictions.append(pred_np)
                
                # Update sequence for next prediction (if multi-step)
                if step < steps_ahead - 1:
                    # Use predicted close price to update sequence
                    new_point = current_sequence[-1].copy()
                    new_point[3] = pred_np[0]  # Update close price
                    current_sequence = np.vstack([current_sequence[1:], new_point])
        
        # Inverse transform predictions (only close prices)
        predictions_rescaled = []
        for pred in predictions:
            # Create dummy array for inverse transform
            dummy = np.zeros((len(pred), 5))
            dummy[:, 3] = pred  # Close price column
            rescaled = self.scaler.inverse_transform(dummy)[:, 3]
            predictions_rescaled.append(rescaled)
        
        return {
            'predictions': predictions_rescaled,
            'prediction_dates': [
                data.index[-1] + timedelta(days=i+1) 
                for i in range(steps_ahead)
            ],
            'model_config': self.config.__dict__,
            'prediction_horizon': self.config.prediction_horizon
        }
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare test data
        X_test, y_test = self.prepare_data(test_data)
        test_loader = self.create_data_loader(X_test, y_test, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals.flatten(), predictions.flatten())
        mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy (for first step prediction)
        if predictions.shape[1] > 0:
            pred_direction = np.sign(predictions[:, 0])
            actual_direction = np.sign(actuals[:, 0])
            directional_accuracy = np.mean(pred_direction == actual_direction)
        else:
            directional_accuracy = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance using gradient-based method"""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        # This is a simplified feature importance calculation
        # In practice, you might want to use more sophisticated methods
        feature_names = ['open', 'high', 'low', 'close', 'volume']
        
        # Calculate average gradient magnitude for each feature
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, self.config.sequence_length, self.config.input_size, 
                                 requires_grad=True, device=self.device)
        
        output = self.model(dummy_input)
        loss = output.sum()
        loss.backward()
        
        # Calculate feature importance as average gradient magnitude
        gradients = dummy_input.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        
        importance_dict = {
            feature_names[i]: float(gradients[i]) 
            for i in range(len(feature_names))
        }
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'training_history': self.training_history
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        
        self.model = LSTMStockModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")