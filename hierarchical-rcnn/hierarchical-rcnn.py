import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalMaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes, class_hierarchy, class_embeddings, num_masks):
        super().__init__()
        self.backbone = backbone
        self.rpn = RPN(backbone.out_channels)
        self.classifier = Classifier(backbone.out_channels, num_classes)
        self.mask = Mask(num_masks)
        
        self.class_hierarchy = class_hierarchy
        self.class_embeddings = nn.Parameter(torch.tensor(class_embeddings))
        
    def forward(self, x):
        features = self.backbone(x)
        proposals, proposal_loss = self.rpn(features, x)
        class_logits, class_probs, bboxes, mask_logits = self.classifier(features, proposals)
        masks = self.mask(features, proposals)
        return class_logits, class_probs, bboxes, mask_logits, masks

class HierarchicalMaskRCNNLoss(nn.Module):
    def __init__(self, hierarchy_loss_weight=1.0):
        super().__init__()
        self.hierarchy_loss_weight = hierarchy_loss_weight
        
    def forward(self, class_logits, class_probs, gt_classes, bboxes, mask_logits, gt_masks):
        class_loss = F.cross_entropy(class_logits, gt_classes)
        bbox_loss = smooth_l1_loss(bboxes, gt_bboxes)
        mask_loss = F.binary_cross_entropy(mask_logits, gt_masks)
        
        hierarchy_loss = self._calculate_hierarchy_loss(class_probs, gt_classes)
        hierarchy_loss *= self.hierarchy_loss_weight
        
        total_loss = class_loss + bbox_loss + mask_loss + hierarchy_loss
        return total_loss
    
    def _calculate_hierarchy_loss(self, class_probs, gt_classes):
        embeddings = self.class_embeddings[gt_classes]
        embedding_distance = torch.mean(torch.norm(embeddings[None, :, :] - embeddings[:, None, :], dim=-1))
        return embedding_distance
