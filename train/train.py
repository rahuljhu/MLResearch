import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import logging
import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_class(module_path: str, class_name: str):
    """Dynamically import a class from a module path."""
    try:
        # Add parent directory to path for relative imports
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to import {class_name} from {module_path}: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required sections
    required_sections = ['model', 'data', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
    
    return config


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Create model instance from configuration."""
    module_path = model_config['module_path']
    class_name = model_config['class_name']
    params = model_config.get('params', {})
    
    logger.info(f"Creating model: {class_name} from {module_path}")
    ModelClass = import_class(module_path, class_name)
    
    model = ModelClass(**params)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_data(data_config: Dict[str, Any]) -> List:
    """Load data from file."""
    data_file = data_config['file_path']
    data_format = data_config.get('format', 'auto')
    
    logger.info(f"Loading data from {data_file}")
    
    # Determine format from file extension if auto
    if data_format == 'auto':
        if data_file.endswith('.json'):
            data_format = 'json'
        elif data_file.endswith('.txt'):
            data_format = 'txt'
        else:
            raise ValueError(f"Cannot auto-detect format for {data_file}")
    
    # Load data based on format
    if data_format == 'json':
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_format == 'txt':
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n')
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    logger.info(f"Loaded {len(data)} data items")
    return data


def create_dataset(data: List, data_config: Dict[str, Any]) -> Dataset:
    """Create dataset instance from data and configuration."""
    if 'dataset_class' in data_config:
        # Custom dataset class specified
        module_path = data_config['dataset_module_path']
        class_name = data_config['dataset_class']
        dataset_params = data_config.get('dataset_params', {})
        
        logger.info(f"Creating custom dataset: {class_name}")
        DatasetClass = import_class(module_path, class_name)
        return DatasetClass(data, **dataset_params)
    
    else:
        # Default simple dataset
        class SimpleDataset(Dataset):
            def __init__(self, data, **kwargs):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return SimpleDataset(data)


def create_dataloader(dataset: Dataset, data_config: Dict[str, Any]) -> DataLoader:
    """Create DataLoader with specified parameters."""
    dataloader_params = {
        'batch_size': data_config.get('batch_size', 16),
        'shuffle': data_config.get('shuffle', True),
        'num_workers': data_config.get('num_workers', 0),
        'drop_last': data_config.get('drop_last', False)
    }
    
    # Add custom collate function if specified
    if 'collate_fn' in data_config:
        module_path = data_config['collate_module_path']
        function_name = data_config['collate_fn']
        collate_fn = import_class(module_path, function_name)
        dataloader_params['collate_fn'] = collate_fn
    
    return DataLoader(dataset, **dataloader_params)


def create_optimizer(model: nn.Module, training_config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_type = training_config.get('optimizer', 'Adam')
    learning_rate = training_config.get('learning_rate', 1e-4)
    optimizer_params = training_config.get('optimizer_params', {})
    
    OptimizerClass = getattr(optim, optimizer_type)
    return OptimizerClass(model.parameters(), lr=learning_rate, **optimizer_params)


def create_loss_function(training_config: Dict[str, Any]) -> nn.Module:
    """Create loss function from configuration."""
    loss_type = training_config.get('loss_function', 'CrossEntropyLoss')
    loss_params = training_config.get('loss_params', {})
    
    # Default params for common losses
    if loss_type == 'CrossEntropyLoss' and 'ignore_index' not in loss_params:
        loss_params['ignore_index'] = 0  # Default PAD token
    
    LossClass = getattr(nn, loss_type)
    return LossClass(**loss_params)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    forward_fn=None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        elif isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
        
        optimizer.zero_grad()
        
        # Forward pass
        if forward_fn:
            # Custom forward function
            loss = forward_fn(model, batch, loss_fn)
        else:
            # Default forward - assume model takes batch and returns logits
            # and loss is computed between logits and targets
            if isinstance(batch, dict):
                outputs = model(**batch)
                # Assume 'targets' key exists for loss computation
                targets = batch.get('targets', batch.get('labels'))
                if targets is None:
                    raise ValueError("No targets found in batch for loss computation")
                loss = loss_fn(outputs, targets)
            else:
                # Assume batch contains (inputs, targets) or similar
                outputs = model(batch)
                # This is a simplified case - you may need to customize
                # based on your specific data format
                loss = loss_fn(outputs, batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress
        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    save_path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generic model training script')
    parser.add_argument('--config', required=True, help='Path to training configuration JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create model
    model = create_model(config['model'])
    
    # Setup device
    device_name = config['training'].get('device', 'auto')
    if device_name == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        logger.info(f"Auto-selected device: {device}")
    else:
        device = torch.device(device_name)
        logger.info(f"Using specified device: {device}")
    
    model = model.to(device)
    
    # Load data and create dataset/dataloader
    data = load_data(config['data'])
    dataset = create_dataset(data, config['data'])
    dataloader = create_dataloader(dataset, config['data'])
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, config['training'])
    loss_fn = create_loss_function(config['training'])
    
    # Import custom forward function if specified
    forward_fn = None
    if 'forward_fn' in config['training']:
        module_path = config['training']['forward_module_path']
        function_name = config['training']['forward_fn']
        forward_fn = import_class(module_path, function_name)
        logger.info(f"Using custom forward function: {function_name}")
    
    # Training loop
    num_epochs = config['training'].get('epochs', 10)
    save_path = config['training'].get('checkpoint_path', 'checkpoint.pt')
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(
            model, dataloader, optimizer, loss_fn, device, epoch, forward_fn
        )
        
        # Save checkpoint every few epochs or at the end
        save_interval = config['training'].get('save_interval', num_epochs)
        if epoch % save_interval == 0 or epoch == num_epochs:
            save_checkpoint(model, optimizer, epoch, avg_loss, config, save_path)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()