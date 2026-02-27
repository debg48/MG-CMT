import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Paths to the datasets
DATASET_2_PATH = Path('data/Dataset of Tuberculosis Chest X-rays Images')
DATASET_3_PATH = Path('data/TB_Chest_Radiography_Database')

# Class definitions for Dataset 2
DS2_CLASSES = {
    'Normal Chest X-rays': 0,
    'TB Chest X-rays': 1
}

# Class definitions for Dataset 3
DS3_CLASSES = {
    'Normal': 0,
    'Tuberculosis': 1
}

def analyze_and_plot_dataset(dataset_path, class_mapping, dataset_name, samples_save_path, dist_save_path):
    print(f"\n{'='*50}")
    print(f"Analyzing {dataset_name} at '{dataset_path}'")
    print(f"{'='*50}")
    
    if not dataset_path.exists():
        print(f"[Error] Dataset directory '{dataset_path}' not found.")
        return

    # 1. Calculate class distribution
    distribution = {}
    sample_images = {} # {class_name: [img_path1, img_path2]}
    
    for class_name, label in class_mapping.items():
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"  [Warning] Class directory '{class_name}' not found.")
            distribution[class_name] = 0
            sample_images[class_name] = []
            continue
            
        extensions = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for ext in extensions:
            img_paths.extend(list(class_dir.glob(ext)))
            
        distribution[class_name] = len(img_paths)
        
        # Select 2 random samples (if available)
        random.shuffle(img_paths)
        sample_images[class_name] = img_paths[:2]

    # Print distribution
    total_images = sum(distribution.values())
    print(f"Total Images: {total_images}")
    for class_name, count in distribution.items():
        print(f"  - {class_name} (Label {class_mapping[class_name]}): {count} images ({count/total_images*100:.1f}%)")

    # 2. Plot Bar Chart for Class Distribution
    plt.figure(figsize=(8, 6))
    classes = list(distribution.keys())
    counts = list(distribution.values())
    
    # Simple bar plot
    bars = plt.bar(classes, counts, color=['#4C72B0', '#DD8452'])
    plt.ylabel('Number of Images', fontsize=12)
    plt.title(f'{dataset_name} Class Distribution', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (total_images*0.01), int(yval), ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(dist_save_path, dpi=150)
    print(f"-> Saved distribution bar plot to '{dist_save_path}'")
    plt.close()

    # 3. Plot 2+2 samples in a row (no titles/captions)
    # We want 2 samples from Class 0, and 2 samples from Class 1 -> 4 images total
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Flatten the sample dictionaries into a list of (path, class_name, label)
    plot_items = []
    for class_name, label in class_mapping.items():
        for img_path in sample_images[class_name]:
            plot_items.append((img_path, class_name, label))
            
    # In case there are not enough images
    for i, ax in enumerate(axes):
        if i < len(plot_items):
            img_path, class_name, label = plot_items[i]
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                ax.set_title(class_name, fontsize=12)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Add some space at the top for titles
    plt.savefig(samples_save_path, dpi=150)
    print(f"-> Saved sample visualization to '{samples_save_path}'")
    plt.close()

if __name__ == '__main__':
    # Set seed for reproducibility of chosen samples
    random.seed(42)
    
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    print("Generating class distributions and sample plots for Dataset 2 and 3...")
    
    # Analyze Dataset 2
    analyze_and_plot_dataset(
        dataset_path=DATASET_2_PATH,
        class_mapping=DS2_CLASSES,
        dataset_name="Dataset 2",
        samples_save_path="results/dataset2_samples.png",
        dist_save_path="results/dataset2_distribution.png"
    )
    
    # Analyze Dataset 3
    analyze_and_plot_dataset(
        dataset_path=DATASET_3_PATH,
        class_mapping=DS3_CLASSES,
        dataset_name="Dataset 3",
        samples_save_path="results/dataset3_samples.png",
        dist_save_path="results/dataset3_distribution.png"
    )
    
    print("\nDone! You can check the generated PNG files for the visual samples and distributions.")
