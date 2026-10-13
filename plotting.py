import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# CIFAR-100 Mapping: Fine Class Index -> Superclass Index (0-19)
FINE_TO_SUPER = np.array([
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5,  1,  8,  6, 14, 14, 11, 12,  8, 15,
   13, 16,  7,  0,  1,  8, 17,  9,  2, 11,
    6, 12,  9, 18,  1, 10, 12,  4, 16,  0,
    9, 17, 13, 14, 13, 16, 12,  9, 18,  1,
   10, 12,  8, 18, 13, 14,  2, 11, 12,  2,
    0,  2, 11, 12,  2, 13, 18,  4, 12, 10
])

SUPERCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2"
]

MODEL_COLORS = {
    "Dummy Baseline": "#7f7f7f",      # Gray
    "HistGradientBoosting": "#9467bd",# Purple
    "Logistic Regression": "#2ca02c", # Green
    "Pre-Tuned": "#1f77b4",           # Blue
    "Tuned": "#d62728"                # Red
}


def plot_curves(history, title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss", color="#1f77b4")
    ax1.plot(history["val_loss"], label="Val Loss", linestyle="--", color="#1f77b4")
    ax1.set_title(f"{title_prefix} Loss")
    ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(history["train_acc"], label="Training", color="#1f77b4")
    ax2.plot(history["val_acc"], label="Validation", linestyle="--", color="#1f77b4")
    ax2.axhline(y=1, color="r", linestyle=":", label=f"Chance ({1:.2f}%)")
    ax2.set_title(f"{title_prefix} Accuracy")
    ax2.legend(); ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.show()


def plot_comparison_curves(pre_history, tuned_history, model_names=["Pre-Tune", "Tuned"]):
    name1, name2 = model_names[0], model_names[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    titles = (
        f"Loss Comparison: {name1} vs. {name2}",
        f"Accuracy Comparison: {name1} vs. {name2}",
    )

    for ax, keys, ylabel, title in (
        (ax1, ("train_loss", "val_loss"), "Loss (Cross-Entropy)", titles[0]),
        (ax2, ("train_acc", "val_acc"), "Accuracy (%)", titles[1]),
    ):
        for hist, color, name in zip(
            (pre_history, tuned_history),
            ("#1f77b4", "#d62728"),
            model_names,
        ):
            ax.plot(
                hist[keys[0]],
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{name} Train",
            )
            ax.plot(
                hist[keys[1]],
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"{name} Validation",
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax2.axhline(
        y=1.0, color="gray", linestyle=":", linewidth=1.5, label="Chance (1.00%)"
    )

    ax1.legend(fontsize=10)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_diagnostics(all_preds, all_targets, images_u8, class_names, title_prefix=""):
    """images_u8: CPU uint8 tensor (N, 3, H, W) of raw test images."""
    
    # Sort fine classes so classes in the same superclass are grouped together
    reorder_idx = np.argsort(FINE_TO_SUPER)
    
    # Map predictions/targets to the reordered indices
    idx_map = {old_idx: new_pos for new_pos, old_idx in enumerate(reorder_idx)}
    reordered_targets = np.array([idx_map[t] for t in all_targets])
    reordered_preds = np.array([idx_map[p] for p in all_preds])

    cm = confusion_matrix(reordered_targets, reordered_preds, labels=np.arange(100))

    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(
        cm, 
        annot=False, 
        cmap="Blues", 
        cbar=True, 
        cbar_kws={"pad": 0.01},
        xticklabels=False, 
        yticklabels=False,
    )

    # Center 20 superclass ticks over each 5-class group (indices: 2.5, 7.5, 12.5, ...)
    tick_positions = np.arange(2.5, 100, 5)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(SUPERCLASS_NAMES, rotation=90, fontsize=10)
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(SUPERCLASS_NAMES, rotation=0, fontsize=10)

    # Draw black lines to visually delimit the 20 superclass boundaries
    for i in range(5, 100, 5):
        ax.axhline(i, color="black", linewidth=0.8, alpha=0.7)
        ax.axvline(i, color="black", linewidth=0.8, alpha=0.7)

    # Format Colorbar Label
    cbar = ax.collections[0].colorbar
    cbar.set_label(
        "Sensitivity (Recall) per True Class", 
        rotation=270, 
        labelpad=20,
        fontsize=15,
    )
    
    plt.title(f"{title_prefix} CIFAR-100 Test Set (Grouped by Superclass)", fontsize=18)
    plt.xlabel("Predicted Superclass", fontsize=14)
    plt.ylabel("True Superclass", fontsize=14)
    
    plt.tight_layout()
    plt.show()

    # Misclassification display code
    wrong = np.where(all_preds != all_targets)[0]
    print(f"\n{title_prefix} Total Misclassified Samples: {len(wrong)} / {len(all_targets)}")

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax_i in enumerate(axes.flat):
        ax_i.axis("off")
        if i < len(wrong):
            idx = wrong[i]
            ax_i.imshow(images_u8[idx].permute(1, 2, 0).numpy())
            ax_i.set_title(f"True: {class_names[all_targets[idx]]}\nPred: {class_names[all_preds[idx]]}", color="red", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(results):
    """Grouped vertical bars: x = Accuracy/Precision/Recall/F1, dynamic bars per model, y = Score (%) 0-100."""
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))

    n_models = len(results)
    # Dynamically scale bar width so all bars fit inside each tick group without touching the next
    w = min(0.2, 0.8 / n_models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for k, r in enumerate(results):
        vals = [r[m] for m in keys]

        # Centered offsets for n_models
        offset = (k - (n_models - 1) / 2.0) * w

        bars = ax.bar(
            x + offset,
            vals,
            w,
            label=r["model"],
            color=MODEL_COLORS.get(r["model"]),
        )
        ax.bar_label(
            bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=11
        )

    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Dummy Classifier (1.0%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title(
        "Test Performance Comparison", fontsize=15, fontweight="bold"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.show()


def plot_time_comparison(results):
    """Grouped horizontal bars: y = Train / Test time, dynamic bars per model, x = seconds."""
    rows = [("Train time", "train_time_min"), ("Test time", "test_time_min")]
    y = np.arange(len(rows))

    n_models = len(results)
    # Scale bar height dynamically so bars don't overlap across groups
    h = min(0.2, 0.7 / n_models)

    fig, ax = plt.subplots(figsize=(10, 5))

    for k, r in enumerate(results):
        vals = [
            (r[key] * 60.0) if r[key] is not None else 0.0 for _, key in rows
        ]

        # Centered offsets for n_models
        offset = (k - (n_models - 1) / 2.0) * h

        bars = ax.barh(
            y + offset,
            vals,
            h,
            label=r["model"],
            color=MODEL_COLORS.get(r["model"]),
        )
        ax.bar_label(
            bars, labels=[f"{v:.2f} s" for v in vals], padding=3, fontsize=9
        )

    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows], fontsize=11)
    ax.invert_yaxis()  # Train on top

    # Separator cleanly centered at y = 0.5 between y=0 and y=1
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_title(
        "Train / Test Time (Tuned train time includes Optuna search)",
        fontsize=13,
        fontweight="bold",
    )
    ax.margins(x=0.15)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.show()