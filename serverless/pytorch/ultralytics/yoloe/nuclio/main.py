import base64
import io
import json

from PIL import Image

from model_handler import ModelHandler


def init_context(context):
    context.user_data.model = ModelHandler()


def handler(context, event):
    data = event.body
    if isinstance(data, bytes):
        data = json.loads(data)

    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
    threshold = float(data.get("threshold", 0.20))
    text_prompts = data.get("text_prompts") or ["car", "truck", "bus", "van"]
    results = context.user_data.model.infer(image, text_prompts, threshold)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
