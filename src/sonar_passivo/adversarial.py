import torch
import torch.nn as nn

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        image: Original input tensor.
        epsilon: Strength of the attack.
        data_grad: Gradient of the loss w.r.t the data.
        
    Returns:
        Perturbed image.
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    # Return the perturbed image
    return perturbed_image

def generate_adversarial_batch(model: nn.Module, data: torch.Tensor, target: torch.Tensor, epsilon: float):
    """
    Generates an adversarial batch for a given input batch.
    """
    data.requires_grad = True
    
    output = model(data)
    # Use model's internal criterion or passed one. Assuming model has criterion.
    # Note: If target is not the prediction, we are forcing it away from the True label
    
    loss = model.criterion(output, target)
    model.zero_grad()
    loss.backward()
    
    data_grad = data.grad.data
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    
    return perturbed_data
