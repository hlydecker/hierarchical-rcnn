import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def validate_hierarchical_mask_rcnn(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, gt_classes, gt_bboxes, gt_masks in val_loader:
            class_logits, class_probs, bboxes, mask_logits, masks = model(images)
            _, pred_classes = torch.max(class_probs, dim=1)
            true_labels.append(gt_classes.detach().cpu().numpy())
            pred_labels.append(pred_classes.detach().cpu().numpy())
            loss = loss_fn(class_logits, class_probs, gt_classes, bboxes, mask_logits, gt_masks)
            total_loss += loss.item()

    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    confusion_matrix = confusion_matrix(true_labels, pred_labels)
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a HierarchicalMaskRCNN model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model state dictionary")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the validation data")
    args = parser.parse_args()

    model = HierarchicalMaskRCNN(backbone, num_classes, class_hierarchy, class_embeddings, num_masks)
    model.load_state_dict(torch.load(args.model_path))
    loss_fn = HierarchicalMaskRCNNLoss()

    val_loader = get_data_loader(args.data_dir, "val")
    validate_hierarchical_mask_rcnn(model, val_loader, loss_fn)
