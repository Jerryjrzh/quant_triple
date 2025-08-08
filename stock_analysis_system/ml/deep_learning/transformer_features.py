"""
Transformer-based Feature Extractor

Advanced transformer model for extracting meaningful features from stock data:
- Multi-head attention mechanisms
- Positional encoding for time series
- Feature extraction and representation learning
- Integration with existing ML pipeline
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer model configuration"""
    input_size: int = 5  # OHLCV
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    sequence_length: int = 60
    output_features: int = 64
    batch_size: int = 32
    learning_rate: float = 0.0001
    epochs: int = 50
    warmup_steps: int = 4000


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class StockTransformerEncoder(nn.Module):
    """Transformer encoder for stock feature extraction"""
    
    def __init__(self, config: TransformerConfig):
        super(StockTransformerEncoder, self).__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_encoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.output_features)
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.output_features)
        
    def forward(self, src, src_mask=None):
        # src: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = src.shape
        
        # Project input to d_model dimensions
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        memory = self.transformer_encoder(src, src_mask)
        # memory: (seq_len, batch_size, d_model)
        
        # Transpose back and take mean over sequence
        memory = memory.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Global average pooling over sequence dimension
        pooled = memory.mean(dim=1)  # (batch_size, d_model)
        
        # Project to output features
        output = self.output_projection(pooled)  # (batch_size, output_features)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output, memory  # Return both pooled features and sequence features


class TransformerFeatureExtractor:
    """Transformer-based feature extractor for stock data"""
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
        
        logger.info(f"Initialized Transformer feature extractor with device: {self.device}")
    
    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for transformer training"""
        features = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in features):
            raise ValueError(f"Data must contain columns: {features}")
        
        # Sort by date
        data = data.sort_index()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(data[features].values)
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - self.config.sequence_length + 1):
            sequences.append(scaled_data[i:(i + self.config.sequence_length)])
        
        return np.array(sequences)
    
    def create_data_loader(self, X: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch data loader"""
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
    
    def create_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Create attention mask for transformer"""
        # For now, we don't mask anything (all positions are valid)
        # In practice, you might want to mask certain positions
        return None
    
    def train_unsupervised(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train transformer using unsupervised learning (reconstruction task)"""
        logger.info("Starting Transformer feature extractor training...")
        
        # Prepare training data
        X_train = self.prepare_data(train_data)
        train_loader = self.create_data_loader(X_train, shuffle=True)
        
        # Prepare validation data
        val_loader = None
        if val_data is not None:
            X_val = self.prepare_data(val_data)
            val_loader = self.create_data_loader(X_val, shuffle=False)
        
        # Initialize model
        self.model = StockTransformerEncoder(self.config).to(self.device)
        
        # Add reconstruction head for unsupervised training
        reconstruction_head = nn.Linear(self.config.output_features, 
                                      self.config.input_size).to(self.device)
        
        # Optimizer with warmup
        optimizer = optim.AdamW(
            list(self.model.parameters()) + list(reconstruction_head.parameters()),
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return (self.config.warmup_steps / step) ** 0.5
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params({
                'd_model': self.config.d_model,
                'nhead': self.config.nhead,
                'num_encoder_layers': self.config.num_encoder_layers,
                'dim_feedforward': self.config.dim_feedforward,
                'dropout': self.config.dropout,
                'sequence_length': self.config.sequence_length,
                'output_features': self.config.output_features,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size
            })
            
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                reconstruction_head.train()
                train_loss = 0.0
                
                for batch_idx, (batch_X,) in enumerate(train_loader):
                    batch_X = batch_X.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Extract features
                    features, sequence_features = self.model(batch_X)
                    
                    # Reconstruct input from features
                    # Use the last time step for reconstruction
                    last_timestep = batch_X[:, -1, :]  # (batch_size, input_size)
                    reconstructed = reconstruction_head(features)
                    
                    # Reconstruction loss
                    loss = criterion(reconstructed, last_timestep)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(reconstruction_head.parameters()),
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    reconstruction_head.eval()
                    
                    with torch.no_grad():
                        for batch_X, in val_loader:
                            batch_X = batch_X.to(self.device)
                            
                            features, _ = self.model(batch_X)
                            last_timestep = batch_X[:, -1, :]
                            reconstructed = reconstruction_head(features)
                            
                            val_loss += criterion(reconstructed, last_timestep).item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.model.state_dict(), 'best_transformer_model.pth')
                
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
                self.model.load_state_dict(torch.load('best_transformer_model.pth'))
            
            # Log model
            mlflow.pytorch.log_model(self.model, "transformer_model")
        
        logger.info("Transformer feature extractor training completed")
        
        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss if val_loader else None,
            'best_val_loss': best_val_loss if val_loader else None,
            'epochs_trained': len(self.training_history)
        }
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from stock data"""
        if self.model is None:
            raise ValueError("Model must be trained before extracting features")
        
        self.model.eval()
        
        # Prepare data
        sequences = self.prepare_data(data)
        data_loader = self.create_data_loader(sequences, shuffle=False)
        
        all_features = []
        all_sequence_features = []
        
        with torch.no_grad():
            for batch_X, in data_loader:
                batch_X = batch_X.to(self.device)
                
                features, sequence_features = self.model(batch_X)
                
                all_features.append(features.cpu().numpy())
                all_sequence_features.append(sequence_features.cpu().numpy())
        
        # Concatenate all features
        pooled_features = np.vstack(all_features)
        sequence_features = np.vstack(all_sequence_features)
        
        return {
            'pooled_features': pooled_features,  # (n_samples, output_features)
            'sequence_features': sequence_features,  # (n_samples, seq_len, d_model)
            'feature_names': [f'transformer_feature_{i}' for i in range(self.config.output_features)],
            'extraction_config': self.config.__dict__
        }
    
    def get_attention_weights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get attention weights for interpretability"""
        if self.model is None:
            raise ValueError("Model must be trained before getting attention weights")
        
        # This is a simplified version - in practice, you'd need to modify
        # the model to return attention weights
        self.model.eval()
        
        # Prepare a single sequence
        sequences = self.prepare_data(data)
        if len(sequences) == 0:
            return {'attention_weights': None}
        
        # Take the first sequence
        sample_sequence = torch.FloatTensor(sequences[0:1]).to(self.device)
        
        with torch.no_grad():
            # For now, return dummy attention weights
            # In a full implementation, you'd modify the transformer to return these
            seq_len = sample_sequence.shape[1]
            attention_weights = torch.ones(self.config.nhead, seq_len, seq_len) / seq_len
        
        return {
            'attention_weights': attention_weights.cpu().numpy(),
            'sequence_length': seq_len,
            'num_heads': self.config.nhead
        }
    
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
        
        self.model = StockTransformerEncoder(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def fine_tune_for_task(self, train_data: pd.DataFrame, labels: np.ndarray, 
                          task_type: str = 'classification') -> Dict[str, Any]:
        """Fine-tune the pre-trained transformer for a specific task"""
        if self.model is None:
            raise ValueError("Model must be pre-trained before fine-tuning")
        
        logger.info(f"Fine-tuning transformer for {task_type} task...")
        
        # Freeze transformer parameters (optional)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add task-specific head
        if task_type == 'classification':
            num_classes = len(np.unique(labels))
            task_head = nn.Sequential(
                nn.Linear(self.config.output_features, self.config.output_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.output_features // 2, num_classes)
            ).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:  # regression
            task_head = nn.Sequential(
                nn.Linear(self.config.output_features, self.config.output_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.output_features // 2, 1)
            ).to(self.device)
            criterion = nn.MSELoss()
        
        # Prepare data
        sequences = self.prepare_data(train_data)
        
        # Create data loader with labels
        X_tensor = torch.FloatTensor(sequences)
        y_tensor = torch.LongTensor(labels) if task_type == 'classification' else torch.FloatTensor(labels)
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Optimizer for task head only
        optimizer = optim.Adam(task_head.parameters(), lr=0.001)
        
        # Fine-tuning loop
        task_head.train()
        self.model.eval()  # Keep transformer frozen
        
        for epoch in range(20):  # Fewer epochs for fine-tuning
            total_loss = 0.0
            
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Extract features (no gradients for transformer)
                with torch.no_grad():
                    features, _ = self.model(batch_X)
                
                # Task-specific prediction
                outputs = task_head(features)
                
                if task_type == 'classification':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            if epoch % 5 == 0:
                logger.info(f"Fine-tuning Epoch {epoch}: Loss: {avg_loss:.6f}")
        
        return {
            'task_head': task_head,
            'fine_tuning_loss': avg_loss,
            'task_type': task_type
        }