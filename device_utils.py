import torch
from typing import Union, Optional


class DeviceManager:
    """Utility class for managing device placement and tensor movement."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize DeviceManager.
        
        Args:
            device: Target device. If None, auto-selects best available device.
        """
        if device is None:
            device = self.get_best_device()
        
        self.device = torch.device(device)
        print(f"DeviceManager initialized with device: {self.device}")
    
    @staticmethod
    def get_best_device() -> str:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def list_available_devices():
        """Print all available devices."""
        print("Available devices:")
        print(f"  CPU: Always available")
        print(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"      {i}: {torch.cuda.get_device_name(i)}")
        print(f"  MPS: {torch.backends.mps.is_available()}")
    
    def to_device(self, *tensors_or_models):
        """
        Move tensors or models to the target device.
        
        Args:
            *tensors_or_models: Variable number of tensors or models to move
            
        Returns:
            Single item if one input, tuple if multiple inputs
        """
        results = []
        for item in tensors_or_models:
            if hasattr(item, 'to'):
                results.append(item.to(self.device))
            else:
                # Handle cases where item might be a list/tuple of tensors
                if isinstance(item, (list, tuple)):
                    moved_items = [self.to_device(sub_item) for sub_item in item]
                    results.append(type(item)(moved_items))
                else:
                    results.append(item)
        
        return results[0] if len(results) == 1 else tuple(results)
    
    def set_device(self, device: Union[str, torch.device]):
        """Change the target device."""
        self.device = torch.device(device)
        print(f"Device changed to: {self.device}")
    
    def get_device_info(self):
        """Get information about current device."""
        info = {
            'device': str(self.device),
            'type': self.device.type
        }
        
        if self.device.type == 'cuda':
            info.update({
                'cuda_version': torch.version.cuda,
                'device_name': torch.cuda.get_device_name(self.device),
                'memory_allocated': torch.cuda.memory_allocated(self.device),
                'memory_reserved': torch.cuda.memory_reserved(self.device)
            })
        elif self.device.type == 'mps':
            info.update({
                'mps_available': torch.backends.mps.is_available(),
                'mps_built': torch.backends.mps.is_built()
            })
        
        return info
    
    def print_device_info(self):
        """Print device information."""
        info = self.get_device_info()
        print(f"\nCurrent Device: {info['device']}")
        print(f"Device Type: {info['type']}")
        
        if info['type'] == 'cuda':
            print(f"CUDA Version: {info.get('cuda_version', 'N/A')}")
            print(f"Device Name: {info.get('device_name', 'N/A')}")
            print(f"Memory Allocated: {info.get('memory_allocated', 0) / 1024**2:.1f} MB")
            print(f"Memory Reserved: {info.get('memory_reserved', 0) / 1024**2:.1f} MB")
        elif info['type'] == 'mps':
            print(f"MPS Available: {info.get('mps_available', False)}")
            print(f"MPS Built: {info.get('mps_built', False)}")


# Convenience functions
def get_device_manager(device: Optional[Union[str, torch.device]] = None) -> DeviceManager:
    """Create a DeviceManager instance."""
    return DeviceManager(device)


def auto_device():
    """Get the best available device."""
    return DeviceManager.get_best_device()


def to_device(device: Union[str, torch.device, DeviceManager], *items):
    """
    Convenience function to move items to device.
    
    Args:
        device: Target device (string, torch.device, or DeviceManager)
        *items: Items to move to device
        
    Returns:
        Moved items
    """
    if isinstance(device, DeviceManager):
        return device.to_device(*items)
    else:
        device_manager = DeviceManager(device)
        return device_manager.to_device(*items)


# Global device manager (optional convenience)
_global_device_manager = None


def set_global_device(device: Union[str, torch.device]):
    """Set a global device manager."""
    global _global_device_manager
    _global_device_manager = DeviceManager(device)
    return _global_device_manager


def get_global_device_manager() -> DeviceManager:
    """Get the global device manager (creates one if doesn't exist)."""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def global_to_device(*items):
    """Move items to the global device."""
    return get_global_device_manager().to_device(*items)


# Example usage and testing
if __name__ == "__main__":
    # List available devices
    DeviceManager.list_available_devices()
    
    # Create device manager (auto-selects best device)
    dm = DeviceManager()
    dm.print_device_info()
    
    # Create some test tensors
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    
    print(f"\nOriginal tensors device: {x.device}")
    
    # Move to device
    x_device, y_device = dm.to_device(x, y)
    print(f"After moving to device: {x_device.device}")
    
    # Test with different devices
    print("\n--- Testing different devices ---")
    
    # Test CPU
    dm_cpu = DeviceManager('cpu')
    x_cpu = dm_cpu.to_device(x_device)
    print(f"CPU device: {x_cpu.device}")
    
    # Test MPS (if available)
    if torch.backends.mps.is_available():
        dm_mps = DeviceManager('mps')
        x_mps = dm_mps.to_device(x_cpu)
        print(f"MPS device: {x_mps.device}")
    
    # Test convenience functions
    print("\n--- Testing convenience functions ---")
    best_device = auto_device()
    print(f"Best device: {best_device}")
    
    # Test global device manager
    set_global_device('cpu')
    x_global = global_to_device(torch.randn(2, 2))
    print(f"Global device tensor: {x_global.device}")