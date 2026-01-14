import os

# Keep tests fast and independent from the heavy TF model load.
os.environ.setdefault("SKIP_MODEL_LOAD", "1")

import io
import numpy as np
from PIL import Image

import app as app_module


class _FakeModel:
    output_shape = (None, 38)

    def predict(self, x):
        batch = x.shape[0]
        out = np.zeros((batch, 38), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def _make_jpeg_bytes():
    image = Image.new("RGB", (224, 224), (0, 255, 0))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_healthz_ok():
    client = app_module.app.test_client()
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"


def test_predict_rejects_missing_file():
    client = app_module.app.test_client()
    res = client.post("/predict_disease", data={})
    assert res.status_code == 400


def test_predict_works_with_fake_model(monkeypatch):
    # Ensure class names exist and align with fake output
    monkeypatch.setattr(app_module, "DISEASE_CLASS_NAMES", [f"Class___{i}" for i in range(38)])
    monkeypatch.setattr(app_module, "disease_model", _FakeModel())
    monkeypatch.setattr(app_module, "preprocess_input", lambda x: x)
    monkeypatch.setattr(app_module, "check_if_plant", lambda img: True)
    monkeypatch.setattr(app_module, "get_fertilizer_recommendation", lambda *args, **kwargs: "OK")

    client = app_module.app.test_client()
    img_buf = _make_jpeg_bytes()

    data = {
        "file": (img_buf, "leaf.jpg"),
        "lat": "18.5204",
        "lon": "73.8567",
    }

    res = client.post("/predict_disease", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    payload = res.get_json()
    assert "disease" in payload
    assert payload["confidence"] == "100.00"
    assert payload["fertilizer"] == "OK"


def test_chat_refuses_empty_message():
    client = app_module.app.test_client()
    res = client.post("/chat", json={"message": ""})
    assert res.status_code == 200
    assert "Please say something" in res.get_json()["reply"]
