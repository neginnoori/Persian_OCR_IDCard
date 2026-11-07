# External components

Model weights and third-party libraries are not bundled. Their upstream terms apply separately.

- Grounding DINO Tiny: https://huggingface.co/IDEA-Research/grounding-dino-tiny
- SAM ViT Base: https://huggingface.co/facebook/sam-vit-base
- Transformers 4.57.6: https://github.com/huggingface/transformers/tree/v4.57.6
- EasyOCR 1.7.2 (Persian/English orientation evidence): https://github.com/JaidedAI/EasyOCR/tree/v1.7.2
- OpenCV (geometry and distributed face cascade): https://github.com/opencv/opencv
- PyTorch CUDA installation profiles: https://pytorch.org/get-started/previous-versions/
- Pillow: https://python-pillow.org/
- NumPy: https://numpy.org/

The package uses the public model IDs above with the providers' default revisions. Weight snapshots
are cached but are not pinned by commit SHA in this release; archive the populated model cache if an
identical offline deployment is required. No PyPI upload or external hosting is part of this delivery.
