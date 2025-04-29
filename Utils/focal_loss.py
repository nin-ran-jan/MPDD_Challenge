# Utils/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation that correctly handles class weighting.

    Args:
        class_weights (torch.Tensor, optional): A manual rescaling weight given to each class.
                                                If given, has to be a Tensor of size C (number of classes).
                                                Higher weights give more importance to under-represented classes.
                                                Defaults to None (no class weighting).
        gamma (float, optional): Focusing parameter. Higher values down-weight easy examples more.
                                 Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Register weights buffer (moves with the model to CPU/GPU)
        # Ensure it's a float tensor if provided
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model predictions, raw logits expected. Shape (N, C).
            targets (torch.Tensor): True labels, integers 0 <= targets < C. Shape (N,).
        """
        # Calculate base cross entropy loss *without reduction*
        # Pass the class weights directly to F.cross_entropy
        # Ensure weights are on the same device as preds
        current_class_weights = None
        if self.class_weights is not None:
             current_class_weights = self.class_weights.to(preds.device) # Move weights to correct device

        # Calculate Cross Entropy loss per sample, applying class weights here
        # reduction='none' is crucial to get per-sample loss before focal modulation
        ce_loss = F.cross_entropy(preds, targets,
                                  weight=current_class_weights,
                                  reduction='none')

        # Calculate pt = exp(-ce_loss). This is the probability of the true class
        pt = torch.exp(-ce_loss)

        # Calculate the focal loss component: (1 - pt)^gamma * ce_loss
        # The 'alpha' balancing from the paper is effectively handled by the 'weight'
        # parameter passed to F.cross_entropy above.
        focal_term = (1 - pt) ** self.gamma
        focal_loss = focal_term * ce_loss

        # Apply the final reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'none', 'mean', or 'sum'.")