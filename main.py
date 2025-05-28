import torch
from src.loader import get_loaders
from src.model import TrafficSignCNN
from src.train import train_model
from src.utils import save_to_csv
def main():
    data_dir = r"C:\Users\prath\Downloads\Road_sign"
    batch_size = 32
    epochs = 15
    learning_rate = 0.001

    train_loader, val_loader, class_names = get_loaders(data_dir, batch_size)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN(num_classes).to(device)

    train_model(model, train_loader, val_loader, device, epochs=epochs, lr=learning_rate)

if __name__ == "__main__":
    main()
