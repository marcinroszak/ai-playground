"""
Prediction script for recognizing digits from images.
Can predict from single images or test images from MNIST dataset.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import DigitClassifier


def load_model(model_path='digit_classifier.pth'):
    """Load a trained model from file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_image(model, device, image_path=None, image_tensor=None):
    """
    Predict digit from an image.
    
    Args:
        model: Trained model
        device: Device to run inference on
        image_path: Path to image file (optional)
        image_tensor: Preprocessed image tensor (optional)
    
    Returns:
        Predicted digit (0-9) and confidence scores
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load image if path is provided
    if image_path:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to MNIST size
        image_tensor = transform(image).unsqueeze(0)
    elif image_tensor is None:
        raise ValueError("Either image_path or image_tensor must be provided")
    
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_digit = predicted.item()
        confidence_score = confidence.item()
    
    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy()
    
    return predicted_digit, confidence_score, all_probs


def visualize_prediction(image_tensor, predicted_digit, confidence, all_probs):
    """Visualize the prediction with the image and confidence scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    axes[0].imshow(image_tensor.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title(f'Predicted: {predicted_digit} (Confidence: {confidence:.2%})')
    axes[0].axis('off')
    
    # Display confidence scores
    axes[1].bar(range(10), all_probs)
    axes[1].set_xlabel('Digit')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Prediction Probabilities')
    axes[1].set_xticks(range(10))
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def test_on_mnist(model_path='digit_classifier.pth', num_samples=10):
    """Test the model on random samples from MNIST test set."""
    print(f'Testing model on {num_samples} random MNIST test images...\n')
    
    # Load model
    model, device = load_model(model_path)
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Get random samples
    indices = torch.randperm(len(test_dataset))[:num_samples]
    
    correct = 0
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(indices):
        image, true_label = test_dataset[sample_idx]
        
        # Predict
        predicted, confidence, _ = predict_image(model, device, image_tensor=image.unsqueeze(0))
        
        # Display
        axes[idx].imshow(image.squeeze().numpy(), cmap='gray')
        color = 'green' if predicted == true_label else 'red'
        axes[idx].set_title(f'True: {true_label}, Pred: {predicted}\n({confidence:.2%})', color=color)
        axes[idx].axis('off')
        
        if predicted == true_label:
            correct += 1
    
    plt.suptitle(f'MNIST Test Samples ({correct}/{num_samples} correct)', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    print(f'Results: {correct}/{num_samples} correct ({100*correct/num_samples:.1f}%)')
    print('Test predictions saved as test_predictions.png')
    
    return correct / num_samples


def predict_from_file(image_path, model_path='digit_classifier.pth'):
    """Predict digit from an image file."""
    print(f'Predicting digit from image: {image_path}\n')
    
    model, device = load_model(model_path)
    predicted, confidence, all_probs = predict_image(model, device, image_path=image_path)
    
    print(f'Predicted digit: {predicted}')
    print(f'Confidence: {confidence:.2%}')
    print('\nAll probabilities:')
    for digit in range(10):
        print(f'  {digit}: {all_probs[digit]:.2%}')
    
    # Load and visualize
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image.resize((28, 28))).unsqueeze(0)
    
    fig = visualize_prediction(image_tensor, predicted, confidence, all_probs)
    plt.savefig('prediction_result.png')
    print('\nPrediction visualization saved as prediction_result.png')


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # If image path is provided
        image_path = sys.argv[1]
        predict_from_file(image_path)
    else:
        # Otherwise test on MNIST samples
        test_on_mnist()


