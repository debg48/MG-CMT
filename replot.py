import yaml
import numpy as np
from pathlib import Path
from utils.visualization import plot_confusion_matrix

def replot_cm(results_path: str, save_path: str, class_names: list, title: str):
    """Load yaml and replot confusion matrix."""
    with open(results_path, 'r') as f:
        data = yaml.unsafe_load(f)
    
    cm = np.array(data['confusion_matrix'])
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(cm, class_names=class_names, save_path=save_path, title=title)
    print(f"Saved {save_path}")

if __name__ == '__main__':
    # Dataset 2
    ds2_dir = "checkpoints/mgm_tb_net_dataset2_dataset2_20260226_171250"
    replot_cm(
        results_path=f"{ds2_dir}/test_results.yaml",
        save_path=f"{ds2_dir}/plots/confusion_matrix_test_original_labels.png",
        class_names=["Normal", "Tuberculosis"],
        title="Test Set Confusion Matrix (Dataset 2)"
    )

    # Dataset 3
    ds3_dir = "checkpoints/mgm_tb_net_dataset3_dataset3_20260228_005840"
    replot_cm(
        results_path=f"{ds3_dir}/test_results.yaml",
        save_path=f"{ds3_dir}/plots/confusion_matrix_test_original_labels.png",
        class_names=["Normal", "Tuberculosis"],
        title="Test Set Confusion Matrix (Dataset 3)"
    )
