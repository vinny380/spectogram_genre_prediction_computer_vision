import os
import random
import shutil

def create_subsampled_dataset(source_dir, target_dir, max_samples_per_class=100):
    """
    Create a subsampled dataset where each class has up to max_samples_per_class samples.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # Get all file names for this class
        all_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        # Randomly select up to max_samples_per_class files
        sampled_files = random.sample(all_files, min(max_samples_per_class, len(all_files)))

        # Copy the selected files to the target directory
        for file_name in sampled_files:
            shutil.copy(os.path.join(class_dir, file_name), os.path.join(target_class_dir, file_name))