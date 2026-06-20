# app/model/onnx_infer.py
import os
import logging
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import onnxruntime as ort

logger = logging.getLogger(__name__)

# --- preprocessing constants ---
IMG = 64
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ----------------- init session -----------------
def init_session(onnx_path: str = "/app/app/model/model.onnx", providers=None) -> Tuple[ort.InferenceSession, str, object]:
    if providers is None:
        providers = ["CPUExecutionProvider"]
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    inp = sess.get_inputs()[0]
    logger.info("ONNX loaded: %s", onnx_path)
    logger.info("ONNX input name=%s shape=%s type=%s", inp.name, inp.shape, inp.type)
    return sess, inp.name, inp.shape

# ----------------- preprocessing -----------------
def preprocess_pil_image(img_pil: Image.Image, img_size: int = IMG) -> np.ndarray:
    img = img_pil.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN[None, None, :]) / STD[None, None, :]
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr.astype(np.float32)

# ----------------- safe runner -----------------
def run_onnx_safe(sess: ort.InferenceSession, input_name: str, batch_array: np.ndarray, chunk_size: int = 32) -> np.ndarray:
    N = batch_array.shape[0]
    model_shape = sess.get_inputs()[0].shape
    first_dim = model_shape[0]
    outputs = []
    # model fixed batch == 1
    if isinstance(first_dim, int) and first_dim == 1:
        for i in range(N):
            single = batch_array[i:i+1]
            out = sess.run(None, {input_name: single})
            outputs.append(out[0])
        return np.vstack(outputs)
    # dynamic batch supported -> chunking
    for i in range(0, N, chunk_size):
        chunk = batch_array[i:i+chunk_size]
        out = sess.run(None, {input_name: chunk})
        outputs.append(out[0])
    return np.vstack(outputs)

# ----------------- inference helpers -----------------
def infer_pil_list(sess: ort.InferenceSession, input_name: str, pil_images: List[Image.Image], chunk_size: int = 32) -> np.ndarray:
    arrs = [preprocess_pil_image(im, IMG) for im in pil_images]
    batch = np.stack(arrs, axis=0)
    logits = run_onnx_safe(sess, input_name, batch, chunk_size=chunk_size)
    # softmax
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs

def infer_full_image_pil(img_pil: Image.Image, sess: ort.InferenceSession, input_name: str,
                         grid_n: int = 19, patch_size: int = IMG, chunk_size: int = 32):
    W, H = img_pil.size
    step_x = W / grid_n; step_y = H / grid_n
    patches = []
    coords = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            x1 = int(round(gx * step_x)); y1 = int(round(gy * step_y))
            x2 = int(round(min(W, (gx+1) * step_x))); y2 = int(round(min(H, (gy+1) * step_y)))
            patch = img_pil.crop((x1, y1, x2, y2)).resize((patch_size, patch_size))
            patches.append(patch)
            coords.append((x1, y1, x2, y2))
    probs = infer_pil_list(sess, input_name, patches, chunk_size=chunk_size)
    return probs, coords

# ----------------- postprocess -----------------
def postprocess_grid(probs: np.ndarray,
                     coords: List[tuple],
                     class_names: List[str] = ["empty", "black", "white"],
                     threshold: float = 0.5,
                     skip_empty: bool = True) -> Dict:
    out = {"cells": [], "grid_shape": None}
    if probs is None or len(probs) == 0:
        return out
    N, C = probs.shape
    use_coords = coords if coords and len(coords) == N else [None] * N
    for i in range(N):
        p = probs[i]
        label = int(np.argmax(p))
        prob = float(p[label])
        # пропускаем пустые, если включён флаг
        if skip_empty and label == 0:
            continue
        label_name = class_names[label] if label < len(class_names) else str(label)
        bbox = use_coords[i]
        cell = {"bbox": bbox, "label": label, "label_name": label_name, "prob": prob}
        if prob < threshold:
            cell["low_confidence"] = True
        out["cells"].append(cell)
    sq = int(np.round(np.sqrt(N)))
    if sq * sq == N:
        out["grid_shape"] = (sq, sq)
    return out


# ----------------- drawing helpers (совместимо с разными Pillow) -----------------
from PIL import ImageDraw, ImageFont
import io

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font):
    try:
        if font is not None and hasattr(font, "getsize"):
            return font.getsize(text)
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return (w, h)
        if hasattr(draw, "textsize"):
            return draw.textsize(text, font=font)
    except Exception:
        pass
    return (len(text) * 6, 11)

def draw_boxes_on_image(pil_img: Image.Image, cells: list, show_label=True, threshold=0.2) -> Image.Image:
    img = pil_img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    colors = {0: (200, 200, 200), 1: (255, 0, 0), 2: (0, 255, 0)}

    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox:
            continue
        prob = float(cell.get("prob", 0.0))
        label = int(cell.get("label", 0))
        name = cell.get("label_name", str(label))

        if prob < threshold:
            continue

        x1, y1, x2, y2 = bbox
        color = colors.get(label, (255, 0, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if show_label:
            text = f"{name} {prob:.2f}"
            tw, th = _measure_text(draw, text, font)
            tx1, ty1 = x1, max(0, y1 - th - 6)
            tx2, ty2 = x1 + tw + 6, ty1 + th + 4
            draw.rectangle([tx1, ty1, tx2, ty2], fill=(0, 0, 0))
            draw.text((tx1 + 3, ty1 + 2), text, fill=(255, 255, 255), font=font)

    return img

def pil_image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
