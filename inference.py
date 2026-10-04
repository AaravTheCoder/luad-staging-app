"""Inference engine for the LUAD staging demo app."""
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from multimodal_sts_model import MultimodalFusionNet, Config

Config.GENOMIC_DIM = 480
Config.FUSION_TYPE = "linear"
Config.FREEZE_ALL_RESNET = True

_model    = None
_pca      = None
_scaler   = None
_gene_names = None


def _load_artifacts():
    global _model, _pca, _scaler, _gene_names
    if _model is not None:
        return

    with open("inference_pca.pkl", "rb") as f:
        _pca = pickle.load(f)
    with open("inference_scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)
    _gene_names = np.load("inference_gene_names.npy", allow_pickle=True)

    _model = MultimodalFusionNet(genomic_dim=480)
    state  = torch.load("best_model.pt", map_location="cpu", weights_only=False)
    _model.load_state_dict(state)
    _model.eval()
    print("[Inference] Artifacts loaded.")


def _process_image(pil_img: Image.Image) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0)   # (1,3,224,224)


def _process_genomic(rna_dict: dict) -> tuple:
    """rna_dict: {gene_name: expression_value}. Returns (tensor, missing_count)."""
    vec = np.zeros(len(_gene_names), dtype=np.float32)
    missing = 0
    for i, g in enumerate(_gene_names):
        if g in rna_dict:
            vec[i] = float(rna_dict[g])
        else:
            missing += 1
    vec_scaled = _scaler.transform(vec.reshape(1, -1))
    vec_pca    = _pca.transform(vec_scaled)[0]
    return torch.FloatTensor(vec_pca).unsqueeze(0), missing  # (1,480)


def _gradcam(model, image_tensor):
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["map"] = out

    def bwd_hook(m, gin, gout):
        gradients["map"] = gout[0]

    h1 = model.image_encoder.layer4.register_forward_hook(fwd_hook)
    h2 = model.image_encoder.layer4.register_full_backward_hook(bwd_hook)

    for p in model.image_encoder.layer4.parameters():
        p.requires_grad_(True)

    logit, _ = model(image_tensor, torch.zeros(1, 480))
    model.zero_grad()
    logit.backward()

    h1.remove()
    h2.remove()

    acts = activations["map"]
    grads = gradients["map"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def predict(pil_img: Image.Image, rna_dict: dict) -> dict:
    _load_artifacts()

    image_tensor   = _process_image(pil_img)
    genomic_tensor, missing = _process_genomic(rna_dict)

    with torch.no_grad():
        logit, _ = _model(image_tensor, genomic_tensor)
        prob = torch.sigmoid(logit).item()

    cam = _gradcam(_model, image_tensor)

    genomic_tensor_grad = genomic_tensor.clone().requires_grad_(True)
    logit2, _ = _model(image_tensor, genomic_tensor_grad)
    logit2.backward()
    attribution = (genomic_tensor_grad.grad * genomic_tensor_grad).squeeze().detach().numpy()

    return {
        "probability":    prob,
        "stage":          "Stage III/IV (Late)" if prob >= 0.5 else "Stage I/II (Early)",
        "confidence":     prob if prob >= 0.5 else 1 - prob,
        "gradcam":        cam,
        "attribution":    attribution,
        "missing_genes":  missing,
        "total_genes":    len(_gene_names),
    }
