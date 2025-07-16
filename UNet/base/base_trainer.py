from abc import ABC, abstractmethod
import torch

class BaseTrainer(ABC):
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, optimizer, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def train_epoch(self):
        """
        Training logic for an epoch
        """
        pass

    @abstractmethod
    def train(self):
        """
        Full training logic
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Validation logic
        """
        pass 