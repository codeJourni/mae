import torch
import timm
from models_mae import mae_vit_base_patch16

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Build a MAE model (encoder + decoder)
    model = mae_vit_base_patch16()  # default MAE ViT-Base config
    model.eval()
    print("Built model with", sum(p.numel() for p in model.parameters()) / 1e6, "M params")

    # Fake 224x224 RGB image batch
    x = torch.randn(2, 3, 224, 224)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

    # Forward pass: MAE returns (loss, pred, mask) during pretraining
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)

    print("Forward pass OK.")
    print("loss:", float(loss))
    print("pred shape:", tuple(pred.shape))
    print("mask shape:", tuple(mask.shape))

if __name__ == "__main__":
    main()