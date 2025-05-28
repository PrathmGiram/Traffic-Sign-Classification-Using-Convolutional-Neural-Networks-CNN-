'''import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, output_dir, split_ratio=0.2, seed=42):
    """
    Splits the dataset into train and val folders (80-20 by default).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    class_names = os.listdir(data_dir)
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get all images in class folder
        images = os.listdir(class_path)
        train_imgs, val_imgs = train_test_split(
            images, test_size=split_ratio, random_state=seed
        )

        # Prepare class folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Copy images
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)

    print(f"✅ Split complete!\nTrain directory: {train_dir}\nValidation directory: {val_dir}")

# Example usage:
# Your original data should be like: gtsrb_all/0/, gtsrb_all/1/, ..., gtsrb_all/42/
original_data_dir = 'gtsrb_all'
split_output_dir = 'data'

split_dataset(original_data_dir, split_output_dir, split_ratio=0.2)'''
