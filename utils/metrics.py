import torch
import torch.nn as nn

class ClassBasedAccuracy(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Use device-compatible initialization
        self.correct = torch.zeros(num_classes, dtype=torch.float)
        self.total = torch.zeros(num_classes, dtype=torch.float)

    def reset(self):
        """Reset the states for the metric."""
        # Use in-place zero filling on the existing tensors
        self.correct.zero_()
        self.total.zero_()

    def update_state(self, y_true, y_pred):
        """Update the metric state with the true and predicted values."""
        # Move tensors to the same device if needed
        if y_true.device != self.correct.device:
            # print("oh shett")
            self.correct = self.correct.to(y_true.device)
            self.total = self.total.to(y_true.device)
        
        for cls in range(self.num_classes):
            mask = (y_true == cls)
    
            self.correct[cls] += torch.sum((y_true[mask] == y_pred[mask]).float()).item()
            self.total[cls] += torch.sum(mask.float()).item()

    def compute(self):
        """Compute the accuracy based on accumulated values."""
        # Handle potential empty classes
        accuracy = torch.where(
            self.total > 0, 
            self.correct / (self.total + 1e-6), 
            torch.zeros_like(self.correct)
        )
        return accuracy

    def forward(self, y_true, y_pred):
        """Calculate and return class-based accuracy."""
        self.update_state(y_true, y_pred)
        return self.compute()