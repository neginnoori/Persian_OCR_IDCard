"""Optional BSRGAN x2 inference adapter for text detection experiments."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from PIL import Image


class BSRGANX2:
    def __init__(self, model_path, source_dir, device='cuda'):
        import torch

        model_path = Path(model_path)
        network_path = Path(source_dir) / 'models' / 'network_rrdbnet.py'
        if not model_path.is_file():
            raise FileNotFoundError(f'BSRGAN x2 weights not found: {model_path}')
        if not network_path.is_file():
            raise FileNotFoundError(f'BSRGAN RRDBNet source not found: {network_path}')
        spec = spec_from_file_location('persian_id_ocr_bsrgan_rrdbnet', network_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load BSRGAN architecture: {network_path}')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.model = module.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2)
        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.torch = torch

    def enhance(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError('BSRGAN expects a PIL image')
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor).squeeze(0).permute(1, 2, 0).clamp(0, 1)
        pixels = (output.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(pixels, mode='RGB')
