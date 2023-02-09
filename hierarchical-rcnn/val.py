import argparse
import torch

def validate_hierarchical_mask_rcnn(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, gt_classes, gt_bboxes, gt_masks in val_loader:
            class_logits, class_probs, bboxes, mask_logits, masks = model(images)
            loss = loss_fn(class_logits, class_probs, gt_classes, bboxes, mask_logits, gt_masks)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")

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
