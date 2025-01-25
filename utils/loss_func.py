import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, gamma = 2, reduction = 'mean'):
        super().__init__()
        self.gamma = gamma 
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1) # Since the logits have a shape of (batch_size, num_classes)

        # Directly calculating log_softmax for numerical stability

        # Using negative log likelihood loss of pytorch to calculate product and summation with negative sign
        loss = F.nll_loss(log_probs, targets)

        return loss
    

class CustomFocalLoss(nn.Module):
    def __init__(self, alpha, gamma = 2, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1) 

        # correct_log_probs = log_probs.gather(dim = 1, index = targets.unsqueeze(1)).squeeze(1)
        correct_log_probs = F.nll_loss(log_probs, targets, reduction='none')
        correct_probs = torch.exp(-correct_log_probs)
        # print("shape of correct_log_probs:", correct_log_probs.shape)
        # print("shape of correct_probs:", correct_probs.shape)

        focal_weight = (1 - correct_probs) ** self.gamma
        # print("shape of focal_weight:", focal_weight.shape)
        # print("Shape of self.alpha", self.alpha.shape)
        # print("self.alpha", self.alpha)

        
        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets]  # Shape: (batch_size,)
            # So that for each training example index, corresponding alpha value is returned 
            # And pytorch tensors are optimized for indexing operations 
            '''
            # alpha_t = torch.tensor([self.alpha[target] for target in targets]) 
            # Also valid but slow
            # Matrix multiplication by one-hot encoding targets also works 
            '''
        else:
            alpha_t = 1.0  # Scalar

        
        # print("Shape of alpha_t:",alpha_t.shape)
        # print("alpha_t", alpha_t)
        # print("type of alpha_t:",alpha_t)
        # Step 6: Compute the focal loss
        # loss = focal_weight * alpha_t * correct_log_probs  # Shape: (batch_size,)
        loss = focal_weight * alpha_t * correct_log_probs
        
        # Step 7: Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction