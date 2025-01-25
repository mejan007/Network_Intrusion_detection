import torch
import torch.nn as nn

class ClassBasedAccuracy(nn.Module):
    def __init__(self, num_classes):
        super(ClassBasedAccuracy, self).__init__()
        self.num_classes = num_classes
        self.correct = torch.zeros(num_classes)
        self.total = torch.zeros(num_classes)

    def reset(self):
        """Reset the states for the metric."""
        self.correct.fill_(0)
        self.total.fill_(0)

    def update_state(self, y_true, y_pred):
        """Update the metric state with the true and predicted values."""
        for cls in range(self.num_classes):
            mask = (y_true == cls)
            self.correct[cls] += torch.sum((y_true[mask] == y_pred[mask])).item()  # Correct predictions
            self.total[cls] += torch.sum(mask).item()  # Total number of instances for class cls

    def compute(self):
        """Compute the accuracy based on accumulated values."""
        accuracy = self.correct / (self.total + 1e-6)  # Avoid division by zero
        return accuracy

    def forward(self, y_true, y_pred):
        """Calculate and return class-based accuracy."""
        self.update_state(y_true, y_pred)
        return self.compute()
