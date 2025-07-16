from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseDataLoader(ABC):
    """
    Base class for all data loaders
    """
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @abstractmethod
    def setup(self):
        """Setup train/val/test data loaders"""
        pass

    def get_train_data_generator(self):
        return self.train_loader

    def get_val_data_generator(self):
        return self.val_loader

    def get_test_data_generator(self):
        return self.test_loader 