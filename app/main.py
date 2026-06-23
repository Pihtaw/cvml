from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import Response, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import io, base64, os, logging
import numpy as np
import onnxruntime as ort
from app.model.onnx_infer import run_onnx_detector, detections_from_onnx_output, bboxs_to_grid


logger = logging.getLogger("app")
app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/app/model/model.onnx")

@app.on_event("startup")
def load_onnx_model():
    # Проверка наличия файла
    if not os.path.exists(MODEL_PATH):
        logger.error("ONNX model not found at %s", MODEL_PATH)
        raise RuntimeError(f"ONNX model not found: {MODEL_PATH}")
    providers = []
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception:
        providers = ['CPUExecutionProvider']
        sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    app.state.onnx_sess = sess
    app.state.onnx_input_name = sess.get_inputs()[0].name
    logger.info("ONNX loaded: %s", MODEL_PATH)
    logger.info("ONNX input name=%s shape=%s", app.state.onnx_input_name, sess.get_inputs()[0].shape)

@app.on_event("shutdown")
def close_onnx():
    try:
        if hasattr(app.state, "onnx_sess"):
            del app.state.onnx_sess
        if hasattr(app.state, "onnx_input_name"):
            del app.state.onnx_input_name
        logger.info("ONNX session cleared from app.state")
    except Exception as e:
        logger.exception("Error during ONNX shutdown: %s", e)

# app.state.onnx_sess = ort.InferenceSession('app/model/model.onnx', providers=[...])
# app.state.onnx_input_name = app.state.onnx_sess.get_inputs()[0].name


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <head><title>ONNX Quick Check</title></head>
      <body>
        <h2>Upload image to run model</h2>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept="image/*">
          <input type="submit" value="Upload and Predict">
        </form>
        <p>Model file: <code>/app/app/model/model.onnx</code></p>
      </body>
    </html>
    """

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    sess = app.state.onnx_sess
    input_name = app.state.onnx_input_name

    outputs = run_onnx_detector(sess, img, input_name, img_size=640)
    boxes, scores, classes = detections_from_onnx_output(outputs, img_size=640, conf_thres=0.25, iou_thres=0.45)

    detections = []
    for b,s,c in zip(boxes, scores, classes):
        detections.append({'bbox':[float(x) for x in b.tolist()], 'score':float(s), 'class':int(c)})

    # масштаб, если нужно (как у вас)
    orig_w, orig_h = img.size
    scale = 1.0
    if (orig_w, orig_h) != (640,640):
        scale = orig_w / 640.0

    annotated = draw_detections_on_pil(img, detections, scale_to_original=scale)

    buf = io.BytesIO()
    annotated.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")



# сопоставление классов -> имена и цвета
CLASS_NAMES = {0: "empty", 1: "black", 2: "white"}
CLASS_COLORS = {0: (200,200,200), 1: (0,0,0), 2: (255,255,255)}  # RGB

def draw_detections_on_pil(img_pil: Image.Image, detections: list, scale_to_original: float = 1.0):
    img = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1,y1,x2,y2 = det['bbox']
        if scale_to_original != 1.0:
            x1 *= scale_to_original; y1 *= scale_to_original
            x2 *= scale_to_original; y2 *= scale_to_original
        cls = int(det.get('class', 0))
        score = det.get('score', 0.0)
        color = CLASS_COLORS.get(cls, (255,0,0))
        thickness = max(1, int(round(min(img.size)/200)))
        for t in range(thickness):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)
        label = f"{CLASS_NAMES.get(cls, str(cls))} {score:.2f}"
        tw, th = _text_size(draw, label, font or ImageFont.load_default())
        tx0 = max(0, x1)
        ty0 = max(0, y1 - th - 4)
        tx1 = min(img.size[0], tx0 + tw + 4)
        ty1 = ty0 + th + 4
        draw.rectangle([tx0, ty0, tx1, ty1], fill=(0,0,0))
        draw.text((tx0 + 2, ty0 + 2), label, fill=(255,255,255), font=font)

    return img

def _text_size(draw, text, font):
    """
    Кросс-версионная обёртка: пытаем draw.textbbox -> font.getsize -> fallback len*6
    Возвращает (width, height)
    """
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        return (w, h)
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text) * 6, 11)