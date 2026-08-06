import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Import the model from your app package
from app.vision.model import SudokuCNN


# UPDATED: Added save_path as a parameter
def train_and_evaluate(data_dir: str, save_path: str):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)

    model = SudokuCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    epochs = 10

    print("--- STARTING TRAINING ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

    # UPDATED: Save to the models directory using the provided path
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}!\n")

    print("--- STARTING EVALUATION ---")
    model.eval()
    correct = 0
    total = 0

    # Create lists to track correct predictions and total images for each of the 10 digits
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Loop through the batch and tally up right/wrong for each specific digit
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    print(f"OVERALL ACCURACY: {100 * correct / total:.2f}% \n")

    print("Accuracy per digit:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Digit {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"Digit {i}: No test images.")


if __name__ == "__main__":
    # Dynamically find the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Construct absolute paths for the data and the models folder
    target_data_dir = os.path.join(project_root, "data")
    target_save_path = os.path.join(project_root, "models", "sudoku_cnn.pth")

    # Ensure the models directory exists before saving to it
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)

    train_and_evaluate(target_data_dir, target_save_path)