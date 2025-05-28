import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.evaluate import evaluate
from src.utils import save_to_csv
import os

def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, log_dir="logs/tensorboard", save_path="outputs/best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
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

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        save_to_csv(epoch, avg_loss, val_acc)

        print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {val_acc:.4f}")

    writer.close()
