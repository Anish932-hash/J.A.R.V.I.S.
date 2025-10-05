"""
J.A.R.V.I.S. Neural Network Manager
Advanced neural network training, management, and visualization system
"""

import sys
import os
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class NeuralNetwork(nn.Module if PYTORCH_AVAILABLE else object):
    """Advanced neural network architecture"""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, activation: str = 'relu'):
        if PYTORCH_AVAILABLE:
            super().__init__()
        else:
            # Fallback for when PyTorch is not available
            self.layers = []

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        if PYTORCH_AVAILABLE:
            # Build layers
            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    self._get_activation(activation)
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, output_size))
            self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()

    def forward(self, x):
        """Forward pass"""
        if PYTORCH_AVAILABLE:
            return self.network(x)
        else:
            # Simple fallback implementation
            return x


class TrainingMetrics:
    """Training metrics tracking"""

    def __init__(self):
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.training_times = []
        self.start_time = None

    def start_epoch(self):
        """Start timing an epoch"""
        self.start_time = time.time()

    def end_epoch(self, loss: float, accuracy: float, val_loss: float = None, val_accuracy: float = None):
        """End timing an epoch and record metrics"""
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.training_times.append(epoch_time)

        self.epoch_losses.append(loss)
        self.epoch_accuracies.append(accuracy)

        if val_loss is not None:
            self.validation_losses.append(val_loss)
        if val_accuracy is not None:
            self.validation_accuracies.append(val_accuracy)

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'epochs_completed': len(self.epoch_losses),
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else None,
            'final_accuracy': self.epoch_accuracies[-1] if self.epoch_accuracies else None,
            'best_accuracy': max(self.epoch_accuracies) if self.epoch_accuracies else None,
            'average_epoch_time': np.mean(self.training_times) if self.training_times else None,
            'total_training_time': sum(self.training_times) if self.training_times else None,
            'loss_history': self.epoch_losses,
            'accuracy_history': self.epoch_accuracies,
            'validation_loss_history': self.validation_losses,
            'validation_accuracy_history': self.validation_accuracies
        }


class NeuralNetworkManager:
    """Advanced neural network training and management system"""

    def __init__(self, development_engine):
        self.development_engine = development_engine
        self.jarvis = development_engine.jarvis if hasattr(development_engine, 'jarvis') else None
        self.logger = logging.getLogger('JARVIS.NeuralNetworkManager')

        # Neural network components
        self.current_network = None
        self.optimizer = None
        self.criterion = None
        self.metrics = TrainingMetrics()

        # Training state
        self.is_training = False
        self.training_thread = None
        self.training_callbacks = []

        # Network configurations
        self.network_configs = {}
        self.training_configs = {}

        # Data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Device
        self.device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu') if PYTORCH_AVAILABLE else None

    async def initialize(self):
        """Initialize neural network manager"""
        try:
            self.logger.info("Initializing Neural Network Manager...")

            if not PYTORCH_AVAILABLE:
                self.logger.warning("PyTorch not available - neural network features will be limited")
                return False

            # Load default configurations
            await self._load_network_configs()

            self.logger.info("Neural Network Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing neural network manager: {e}")
            return False

    async def _load_network_configs(self):
        """Load network configurations"""
        try:
            # Default network configurations
            self.network_configs = {
                'command_classifier': {
                    'input_size': 768,  # BERT embedding size
                    'hidden_sizes': [512, 256, 128],
                    'output_size': 50,  # Number of command classes
                    'activation': 'relu'
                },
                'intent_recognizer': {
                    'input_size': 300,  # Word embedding size
                    'hidden_sizes': [256, 128],
                    'output_size': 20,  # Number of intents
                    'activation': 'tanh'
                },
                'sentiment_analyzer': {
                    'input_size': 768,
                    'hidden_sizes': [256, 128],
                    'output_size': 3,  # Positive, negative, neutral
                    'activation': 'relu'
                }
            }

            # Default training configurations
            self.training_configs = {
                'default': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'validation_split': 0.2,
                    'early_stopping_patience': 10,
                    'optimizer': 'adam',
                    'loss_function': 'cross_entropy'
                },
                'fast_training': {
                    'learning_rate': 0.01,
                    'batch_size': 64,
                    'epochs': 50,
                    'validation_split': 0.1,
                    'early_stopping_patience': 5,
                    'optimizer': 'adam',
                    'loss_function': 'cross_entropy'
                }
            }

        except Exception as e:
            self.logger.error(f"Error loading network configs: {e}")

    async def create_network(self, config_name: str, custom_config: Dict[str, Any] = None) -> bool:
        """Create a neural network"""
        try:
            if not PYTORCH_AVAILABLE:
                self.logger.error("PyTorch not available")
                return False

            # Get configuration
            config = self.network_configs.get(config_name, {}).copy()
            if custom_config:
                config.update(custom_config)

            if not config:
                self.logger.error(f"Network configuration '{config_name}' not found")
                return False

            # Create network
            self.current_network = NeuralNetwork(
                input_size=config['input_size'],
                hidden_sizes=config['hidden_sizes'],
                output_size=config['output_size'],
                activation=config.get('activation', 'relu')
            ).to(self.device)

            self.logger.info(f"Created neural network: {config_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating network: {e}")
            return False

    async def prepare_data(self, data: np.ndarray, labels: np.ndarray, config_name: str = 'default') -> bool:
        """Prepare data for training"""
        try:
            if not PYTORCH_AVAILABLE or not SKLEARN_AVAILABLE:
                self.logger.error("Required libraries not available")
                return False

            config = self.training_configs.get(config_name, self.training_configs['default'])

            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                data, labels, test_size=config['validation_split'] * 2, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )

            # Normalize data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.LongTensor(y_test)

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)

            self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            self.test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

            self.logger.info(f"Prepared data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
            return True

        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return False

    async def setup_training(self, config_name: str = 'default') -> bool:
        """Setup training components"""
        try:
            if not PYTORCH_AVAILABLE or not self.current_network:
                return False

            config = self.training_configs.get(config_name, self.training_configs['default'])

            # Setup optimizer
            if config['optimizer'] == 'adam':
                self.optimizer = optim.Adam(self.current_network.parameters(), lr=config['learning_rate'])
            elif config['optimizer'] == 'sgd':
                self.optimizer = optim.SGD(self.current_network.parameters(), lr=config['learning_rate'])
            else:
                self.optimizer = optim.Adam(self.current_network.parameters(), lr=config['learning_rate'])

            # Setup loss function
            if config['loss_function'] == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss()
            elif config['loss_function'] == 'mse':
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

            self.logger.info("Training setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Error setting up training: {e}")
            return False

    def start_training(self, epochs: int = None, callback: callable = None) -> str:
        """Start training in background thread"""
        try:
            if self.is_training:
                return "Training already in progress"

            if not self.current_network or not self.train_loader:
                return "Network or data not prepared"

            if callback:
                self.training_callbacks.append(callback)

            self.is_training = True
            self.metrics = TrainingMetrics()

            training_config = self.training_configs.get('default', {})
            epochs = epochs or training_config.get('epochs', 100)

            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(epochs,),
                daemon=True
            )
            self.training_thread.start()

            self.logger.info(f"Started training for {epochs} epochs")
            return f"Training started for {epochs} epochs"

        except Exception as e:
            self.logger.error(f"Error starting training: {e}")
            return f"Error starting training: {e}"

    def _training_loop(self, epochs: int):
        """Main training loop"""
        try:
            if not PYTORCH_AVAILABLE:
                return

            best_accuracy = 0.0
            patience_counter = 0
            config = self.training_configs.get('default', {})
            patience = config.get('early_stopping_patience', 10)

            for epoch in range(epochs):
                if not self.is_training:
                    break

                self.metrics.start_epoch()

                # Training phase
                self.current_network.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_X, batch_y in self.train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.current_network(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                epoch_loss /= len(self.train_loader)
                epoch_accuracy = correct / total

                # Validation phase
                val_loss, val_accuracy = self._validate()

                self.metrics.end_epoch(epoch_loss, epoch_accuracy, val_loss, val_accuracy)

                # Early stopping
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint(epoch + 1, best_accuracy)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                # Call callbacks
                for callback in self.training_callbacks:
                    try:
                        callback(epoch + 1, epoch_loss, epoch_accuracy, val_loss, val_accuracy)
                    except Exception as e:
                        self.logger.error(f"Error in training callback: {e}")

                self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            self.is_training = False
            self.logger.info("Training completed")

        except Exception as e:
            self.logger.error(f"Error in training loop: {e}")
            self.is_training = False

    def _validate(self) -> Tuple[float, float]:
        """Validate the model"""
        if not PYTORCH_AVAILABLE or not self.val_loader:
            return 0.0, 0.0

        try:
            self.current_network.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = self.current_network(batch_X)
                    loss = self.criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_loss /= len(self.val_loader)
            val_accuracy = correct / total

            return val_loss, val_accuracy

        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            return 0.0, 0.0

    def stop_training(self) -> str:
        """Stop training"""
        try:
            if not self.is_training:
                return "No training in progress"

            self.is_training = False
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=5.0)

            self.logger.info("Training stopped")
            return "Training stopped"

        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
            return f"Error stopping training: {e}"

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'current_epoch': len(self.metrics.epoch_losses),
            'metrics': self.metrics.get_stats(),
            'device': str(self.device) if self.device else 'cpu',
            'network_info': {
                'input_size': getattr(self.current_network, 'input_size', None) if self.current_network else None,
                'output_size': getattr(self.current_network, 'output_size', None) if self.current_network else None,
                'hidden_sizes': getattr(self.current_network, 'hidden_sizes', None) if self.current_network else None
            } if self.current_network else None
        }

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if not PYTORCH_AVAILABLE or not self.current_network:
                return np.array([])

            self.current_network.eval()

            # Normalize data (assuming same scaler as training)
            # For simplicity, we'll assume data is already preprocessed

            data_tensor = torch.FloatTensor(data).to(self.device)

            with torch.no_grad():
                outputs = self.current_network(data_tensor)
                _, predicted = torch.max(outputs.data, 1)

            return predicted.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return np.array([])

    def _save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint"""
        try:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models')
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_acc_{accuracy:.4f}.pth')

            if PYTORCH_AVAILABLE:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.current_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy,
                    'timestamp': datetime.now().isoformat()
                }, checkpoint_path)

            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint"""
        try:
            if not PYTORCH_AVAILABLE:
                return False

            checkpoint = torch.load(checkpoint_path)
            self.current_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False

    def reset_network(self) -> bool:
        """Reset the current network"""
        try:
            if self.current_network:
                # Reinitialize network parameters
                if PYTORCH_AVAILABLE:
                    for layer in self.current_network.modules():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()

                self.metrics = TrainingMetrics()
                self.logger.info("Network reset")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error resetting network: {e}")
            return False

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        try:
            if not self.current_network:
                return {}

            total_params = sum(p.numel() for p in self.current_network.parameters()) if PYTORCH_AVAILABLE else 0
            trainable_params = sum(p.numel() for p in self.current_network.parameters() if p.requires_grad) if PYTORCH_AVAILABLE else 0

            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'network_architecture': {
                    'input_size': getattr(self.current_network, 'input_size', None),
                    'hidden_sizes': getattr(self.current_network, 'hidden_sizes', None),
                    'output_size': getattr(self.current_network, 'output_size', None),
                    'activation': getattr(self.current_network, 'activation', None)
                },
                'device': str(self.device) if self.device else 'cpu',
                'pytorch_available': PYTORCH_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE
            }

        except Exception as e:
            self.logger.error(f"Error getting network stats: {e}")
            return {}

    async def shutdown(self):
        """Shutdown neural network manager"""
        try:
            self.stop_training()
            self.logger.info("Neural Network Manager shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down neural network manager: {e}")