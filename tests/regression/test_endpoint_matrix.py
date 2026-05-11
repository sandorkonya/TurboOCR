"""Endpoint x filetype regression matrix.

Hits every public endpoint (HTTP + gRPC) with every supported image filetype,
then re-runs each (endpoint, filetype) pair under concurrent load to catch the
class of bugs where decoder state corruption only appears after a few requests
(e.g. the gRPC+JPEG CUDA fault).

Failures point at the exact (endpoint, filetype) pair via parametrize ids.

Fast mode (commit gate): MATRIX_FAST=1 lowers concurrency to 2 and skips slow
filetypes (TIFF, WebP). Full mode (CI gate): default, concurrency 20.
"""

import base64
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import requests
from PIL import Image, ImageDraw

GRPC_GENERATED = Path(__file__).resolve().parent.parent / "_grpc_generated"
sys.path.insert(0, str(GRPC_GENERATED))

try:
    import grpc
    import ocr_pb2
    import ocr_pb2_grpc
    _HAS_GRPC = True
except ImportError:
    _HAS_GRPC = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


EXPECTED_TEXT = "Regression OK 12345"
EXPECTED_TOKENS = ("regression", "ok", "12345", "regress", "12345")

FAST = os.environ.get("MATRIX_FAST") == "1"
CONCURRENCY = 2 if FAST else int(os.environ.get("MATRIX_CONCURRENCY", "20"))

IMAGE_FILETYPES = ["jpeg", "png", "bmp", "tiff", "webp"]
if FAST:
    IMAGE_FILETYPES = [ft for ft in IMAGE_FILETYPES if ft not in ("tiff", "webp")]

HTTP_IMAGE_ENDPOINTS = ["raw", "batch", "pixels", "base64"]
GRPC_IMAGE_RPCS = ["Recognize", "RecognizeBatch"]


def _font():
    from PIL import ImageFont
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, 28)
    return ImageFont.load_default()


@pytest.fixture(scope="session")
def matrix_image():
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 100), EXPECTED_TEXT, fill="black", font=_font())
    return img


@pytest.fixture(scope="session")
def encoded_images(matrix_image, tmp_path_factory):
    """Encode the fixture image into every required filetype.

    Returns a dict[ft] -> bytes. Written to a tmp dir so the bytes are also
    inspectable on disk for debugging.
    """
    out_dir = tmp_path_factory.mktemp("matrix_fixtures")
    encoded = {}
    for ft in IMAGE_FILETYPES:
        path = out_dir / f"sample.{ft}"
        save_kwargs = {}
        fmt = ft.upper()
        if ft == "jpeg":
            save_kwargs["quality"] = 90
        matrix_image.save(path, format=fmt, **save_kwargs)
        encoded[ft] = path.read_bytes()
    return encoded


@pytest.fixture(scope="session")
def matrix_pdf_bytes():
    if not _HAS_REPORTLAB:
        return None
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 24)
    c.drawString(100, 700, EXPECTED_TEXT)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def grpc_channel(grpc_target):
    if not _HAS_GRPC:
        pytest.skip("gRPC stubs not available")
    channel = grpc.insecure_channel(grpc_target)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError:
        pytest.skip(f"gRPC server not reachable at {grpc_target}")
    return channel


def _content_type(filetype):
    return {
        "jpeg": "image/jpeg",
        "png": "image/png",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "webp": "image/webp",
    }[filetype]


def _bgr_bytes(filetype, raw_bytes):
    """Decode any filetype to raw BGR bytes for the /ocr/pixels endpoint."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    import numpy as np
    arr = np.asarray(img)
    bgr = arr[:, :, ::-1].copy()
    return bgr.tobytes(), arr.shape[1], arr.shape[0], 3


def _texts_from_results(results):
    return " ".join(r.get("text", "") for r in results).lower()


def _assert_text_present(blob, where):
    low = blob.lower()
    assert any(tok in low for tok in EXPECTED_TOKENS), (
        f"{where}: no expected token found in detected text: {blob!r}"
    )


def _http_call(endpoint, server_url, filetype, raw_bytes):
    if endpoint == "raw":
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=raw_bytes,
            headers={"Content-Type": _content_type(filetype)},
            timeout=30,
        )
        assert r.status_code == 200, f"/ocr/raw {filetype}: {r.status_code} {r.text[:200]}"
        data = r.json()
        assert "results" in data
        _assert_text_present(_texts_from_results(data["results"]), f"/ocr/raw {filetype}")
        return data
    if endpoint == "batch":
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64]},
            timeout=30,
        )
        assert r.status_code == 200, f"/ocr/batch {filetype}: {r.status_code} {r.text[:200]}"
        data = r.json()
        assert data.get("batch_results"), f"/ocr/batch {filetype}: empty batch_results"
        _assert_text_present(
            _texts_from_results(data["batch_results"][0].get("results", [])),
            f"/ocr/batch {filetype}",
        )
        return data
    if endpoint == "base64":
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": b64},
            timeout=30,
        )
        assert r.status_code == 200, f"/ocr (base64) {filetype}: {r.status_code} {r.text[:200]}"
        data = r.json()
        assert "results" in data, f"/ocr (base64) {filetype}: no results key"
        _assert_text_present(_texts_from_results(data["results"]), f"/ocr (base64) {filetype}")
        return data
    if endpoint == "pixels":
        bgr, w, h, ch = _bgr_bytes(filetype, raw_bytes)
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=bgr,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
                "Content-Type": "application/octet-stream",
            },
            timeout=30,
        )
        assert r.status_code == 200, f"/ocr/pixels {filetype}: {r.status_code} {r.text[:200]}"
        data = r.json()
        _assert_text_present(_texts_from_results(data.get("results", [])), f"/ocr/pixels {filetype}")
        return data
    raise AssertionError(f"unknown endpoint {endpoint!r}")


def _grpc_call(rpc, channel, filetype, raw_bytes):
    stub = ocr_pb2_grpc.OCRServiceStub(channel)
    if rpc == "Recognize":
        resp = stub.Recognize(ocr_pb2.OCRRequest(image=raw_bytes), timeout=30)
        if resp.json_response:
            data = json.loads(resp.json_response)
            text = _texts_from_results(data.get("results", []))
        else:
            text = " ".join(r.text for r in resp.results)
        _assert_text_present(text, f"gRPC Recognize {filetype}")
        return resp
    if rpc == "RecognizeBatch":
        resp = stub.RecognizeBatch(ocr_pb2.OCRBatchRequest(images=[raw_bytes]), timeout=30)
        assert resp.batch_results, f"gRPC RecognizeBatch {filetype}: empty batch"
        first = resp.batch_results[0]
        if first.json_response:
            data = json.loads(first.json_response)
            text = _texts_from_results(data.get("results", []))
        else:
            text = " ".join(r.text for r in first.results)
        _assert_text_present(text, f"gRPC RecognizeBatch {filetype}")
        return resp
    raise AssertionError(f"unknown rpc {rpc!r}")


def _stress(call_fn, n):
    """Run call_fn n times concurrently. Every call must succeed."""
    errors = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(call_fn) for _ in range(n)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                errors.append(repr(e))
    assert not errors, f"{len(errors)}/{n} concurrent requests failed:\n" + "\n".join(errors[:5])


@pytest.mark.parametrize("filetype", IMAGE_FILETYPES)
@pytest.mark.parametrize("endpoint", HTTP_IMAGE_ENDPOINTS)
def test_http_image(endpoint, filetype, server_url, encoded_images):
    raw = encoded_images[filetype]
    _http_call(endpoint, server_url, filetype, raw)
    _stress(lambda: _http_call(endpoint, server_url, filetype, raw), CONCURRENCY)


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed (needed for PDF fixture)")
def test_http_pdf(server_url, matrix_pdf_bytes):
    def call():
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=matrix_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=60,
        )
        assert r.status_code == 200, f"/ocr/pdf: {r.status_code} {r.text[:200]}"
        data = r.json()
        assert data.get("pages"), "/ocr/pdf: no pages"
        joined = " ".join(
            _texts_from_results(p.get("results", [])) for p in data["pages"]
        )
        _assert_text_present(joined, "/ocr/pdf")
    call()
    _stress(call, CONCURRENCY)


@pytest.mark.skipif(not _HAS_GRPC, reason="gRPC stubs not available")
@pytest.mark.parametrize("filetype", IMAGE_FILETYPES)
@pytest.mark.parametrize("rpc", GRPC_IMAGE_RPCS)
def test_grpc_image(rpc, filetype, grpc_channel, encoded_images):
    raw = encoded_images[filetype]
    _grpc_call(rpc, grpc_channel, filetype, raw)
    _stress(lambda: _grpc_call(rpc, grpc_channel, filetype, raw), CONCURRENCY)


@pytest.mark.skipif(not _HAS_GRPC, reason="gRPC stubs not available")
@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed (needed for PDF fixture)")
def test_grpc_pdf(grpc_channel, matrix_pdf_bytes):
    def call():
        stub = ocr_pb2_grpc.OCRServiceStub(grpc_channel)
        resp = stub.RecognizePDF(ocr_pb2.OCRPDFRequest(pdf_data=matrix_pdf_bytes), timeout=60)
        assert resp.pages, "gRPC RecognizePDF: no pages"
        texts = []
        for p in resp.pages:
            if p.json_response:
                data = json.loads(p.json_response)
                texts.append(_texts_from_results(data.get("results", [])))
            else:
                texts.append(" ".join(r.text for r in p.results))
        _assert_text_present(" ".join(texts), "gRPC RecognizePDF")
    call()
    _stress(call, CONCURRENCY)
