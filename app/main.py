from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.responses import Response
import io
from app.model.onnx_infer import (
    init_session,
    infer_full_image_pil,
    postprocess_grid,
    draw_boxes_on_image,
    pil_image_to_bytes,
)

from PIL import Image
import logging
from app.model.onnx_infer import init_session, infer_full_image_pil, postprocess_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="ONNX Quick Check")

# initialize session at startup
try:
    sess, input_name, input_shape = init_session("/app/app/model/model.onnx")
    app.state.onnx_sess = sess
    app.state.onnx_input_name = input_name
    logger.info("Model loaded, input shape: %s", input_shape)
except Exception as e:
    logger.exception("Model init failed: %s", e)
    app.state.onnx_sess = None
    app.state.onnx_input_name = None

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
async def predict(file: UploadFile = File(...)):
    if app.state.onnx_sess is None:
        return JSONResponse({"error":"Model not loaded on server"}, status_code=500)
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    probs, coords = infer_full_image_pil(img, app.state.onnx_sess, app.state.onnx_input_name, grid_n=19, patch_size=64, chunk_size=64)
    grid = postprocess_grid(probs, coords, class_names=["empty","black","white"], threshold=0.0, skip_empty=True)

    drawn = draw_boxes_on_image(img, grid["cells"], show_label=True, threshold=0.2)
    img_bytes = pil_image_to_bytes(drawn, fmt="PNG")
    return Response(content=img_bytes, media_type="image/png")

