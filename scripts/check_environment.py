"""Check the image-first ML environment without downloading model weights."""

import importlib
import platform
from importlib.metadata import version


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "timm": "timm",
    "faiss-cpu": "faiss",
    "pytorch-metric-learning": "pytorch_metric_learning",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}


def main() -> None:
    print(f"Python: {platform.python_version()}")
    for package, module in PACKAGES.items():
        importlib.import_module(module)
        print(f"{package}: {version(package)}")

    import torch
    import torchvision

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9])
    kept = torchvision.ops.nms(boxes, scores, 0.5)
    if kept.tolist() != [0]:
        raise RuntimeError("torchvision NMS smoke check failed.")
    print(f"PyTorch device: cpu (CUDA available: {torch.cuda.is_available()})")
    print("Environment check passed, including torchvision CPU operators.")


if __name__ == "__main__":
    main()
