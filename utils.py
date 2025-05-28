import os
import csv

def save_to_csv(epoch, train_loss, val_acc):
    with open("training_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_acc])