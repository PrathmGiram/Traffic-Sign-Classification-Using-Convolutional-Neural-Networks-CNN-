import os
import shutil

val_path = r"C:\Users\prath\Downloads\Road_sign\val"
new_val_path = r"C:\Users\prath\Downloads\Road_sign\val_fixed"

# Make new directory structure
for img in os.listdir(val_path):
    if img.endswith(('.png', '.jpg')):
        class_id = img.split('_')[0]  # Change this logic based on filename pattern
        class_folder = os.path.join(new_val_path, class_id)
        os.makedirs(class_folder, exist_ok=True)
        shutil.move(os.path.join(val_path, img), os.path.join(class_folder, img))

print("Folder restructuring done.")