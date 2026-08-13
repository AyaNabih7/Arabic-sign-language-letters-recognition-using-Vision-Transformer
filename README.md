# Arabic Sign Language Letters Recognition — Vision Transformer (ASL-ViT)

This repository contains the code, training scripts, and supporting material for the paper:

Alnabih, A.F., Maghari, A.Y. *Arabic sign language letters recognition using Vision Transformer*. Multimedia Tools and Applications, 83, 81725–81739 (2024). https://doi.org/10.1007/s11042-024-18681-3

The trained model is published on Hugging Face: https://huggingface.co/AyaF/ASL-Vit

## Overview

This project develops a Vision Transformer (ViT)-based image-classification system to recognize Arabic sign language letters from hand images. The repository includes dataset preparation and preprocessing, model definition and training code, evaluation scripts, and utilities for inference and visualization.

Key contributions:
- Adapting the Vision Transformer architecture to the Arabic sign language letters recognition task.
- A full training and evaluation pipeline with preprocessing, augmentation, and reporting.
- Public release of the pretrained model on Hugging Face for easy inference and fine-tuning.

For full experimental details, metrics, and analysis, please consult the published paper (DOI above).

## Repository structure (typical)
- data/             — dataset storage or scripts/instructions to download the dataset
- notebooks/        — exploratory notebooks and visualizations
- src/              — model, training, and evaluation code
- scripts/          — convenience scripts for training / evaluation / inference
- requirements.txt  — Python dependencies
- README.md         — this file

Adjust the paths above to match the repository if they differ.

## Requirements

Recommended environment:
- Python 3.8+
- CUDA-enabled GPU (for training)
- PyTorch
- torchvision
- transformers (for Hugging Face model loading)
- timm (optional, for ViT utilities)
- albumentations or torchvision transforms
- scikit-learn, numpy, pandas, matplotlib, opencv-python
- tqdm, einops

Install dependencies:
```
pip install -r requirements.txt
```
Or for inference only:
```
pip install torch torchvision transformers pillow
```

## Quick demo — inference with the Hugging Face model

The pretrained model and processor are available at: https://huggingface.co/AyaF/ASL-Vit

Example (PyTorch + Hugging Face transformers):

```python
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "AyaF/ASL-Vit"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

# Load an example image (RGB)
image = Image.open("path/to/your/hand_image.jpg").convert("RGB")

# Preprocess and forward
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predicted_label_id = int(logits.argmax(-1))
predicted_label = model.config.id2label[predicted_label_id]

print("Predicted class:", predicted_label)
```

Note: follow any preprocessing and label mapping described on the model card at Hugging Face.

## How to reproduce experiments

1. Prepare the dataset and the train/validation/test splits as described in the paper (or in data/ if included).
2. Create a virtual environment and install dependencies.
3. Inspect the training script (e.g., `src/train.py` or `scripts/train.sh`) and configuration (CLI args or config files).
4. Use the same random seed, optimizer, learning-rate schedule, augmentations, batch size, and training epochs as reported in the paper to reproduce reported results.
5. Train on a GPU-enabled machine and save checkpoints. Use the included evaluation script to compute metrics and generate confusion matrices.

Exact hyperparameters, augmentations, and ablation settings are provided in the published article — consult it for precise values.

## Evaluation & Results

Evaluation methodology and metrics (accuracy, per-class performance, confusion matrices) are detailed in the paper. The Hugging Face model card may also include evaluation summaries and notes.

## Dataset & Ethics

- The dataset description and the data collection/annotation process are provided in the paper. Please follow any licensing or usage restrictions indicated there.
- Be mindful of privacy, consent, and ethical considerations when using image data of people and signers. Use and share the dataset and models responsibly.

## Cite this work

If you use this repository, the model, or the results in your research, please cite:

BibTeX:
```bibtex
@article{Alnabih2024ASLViT,
  title={Arabic sign language letters recognition using Vision Transformer},
  author={Alnabih, A. F. and Maghari, A. Y.},
  journal={Multimedia Tools and Applications},
  volume={83},
  pages={81725--81739},
  year={2024},
  doi={10.1007/s11042-024-18681-3},
  url={https://doi.org/10.1007/s11042-024-18681-3}
}
```

## Model (Hugging Face)

Pretrained model and model card: https://huggingface.co/AyaF/ASL-Vit

Please check the model card for details about intended use, limitations, and preprocessing recommendations.

## License

Add a LICENSE file to indicate usage permissions (e.g., MIT, Apache-2.0). If you tell me which license you prefer, I can add a suitable LICENSE file.

## Contact

Author: A. F. Alnabih (Aya Nabih)  
Paper: https://doi.org/10.1007/s11042-024-18681-3  
Hugging Face model: https://huggingface.co/AyaF/ASL-Vit

