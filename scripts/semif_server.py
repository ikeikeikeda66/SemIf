#!/usr/bin/env python3
"""SemIf Resident HTTP Server for Apple Silicon (MPS / float16).

Provides high-throughput, low-latency REST endpoints for Claude Desktop,
Claude Code, agents, and applications.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semif_phase1.core import load_causal_model, validate_row
from semif_phase1.direct import score as direct_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("semif-server")

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

model = None
tokenizer = None
metadata = None
model_lock = threading.Lock()
server_ready = False
model_load_error = None


class SemIfRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for SemIf semantic decision endpoints."""

    def _send_json(self, status: int, data: dict):
        response_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            if not server_ready:
                status = HTTPStatus.SERVICE_UNAVAILABLE
                response = {
                    "status": "loading",
                    "ready": False,
                    "error": model_load_error,
                }
            else:
                status = HTTPStatus.OK
                response = {
                    "status": "ok",
                    "ready": True,
                    "model": metadata,
                }
            self._send_json(status, response)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found", "path": self.path})

    def do_POST(self):
        global model, tokenizer, metadata

        if not server_ready:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Model is not ready", "details": model_load_error},
            )
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing request body"})
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {e}"})
            return

        if self.path == "/v1/decide":
            # High-level convenient decision API
            # Payload format:
            # {
            #   "state": "...",
            #   "question": "...",
            #   "options": [{"id": "yes", "description": "Yes"}, ...] or ["yes", "no"]
            # }
            state = payload.get("state")
            question = payload.get("question")
            options = payload.get("options")

            if state is None or question is None or not options:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Missing required fields: state, question, options"},
                )
                return

            formatted_options = []
            for item in options:
                if isinstance(item, str):
                    formatted_options.append({"id": item, "description": item})
                elif isinstance(item, dict) and "id" in item:
                    formatted_options.append({
                        "id": item["id"],
                        "description": item.get("description", item["id"]),
                    })
                else:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": f"Invalid option format: {item}"},
                    )
                    return

            row = {
                "id": payload.get("id", f"decide-{int(time.time()*1000)}"),
                "state": state,
                "question": question,
                "options": formatted_options,
            }

            try:
                validate_row(row)
            except Exception as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Validation error: {e}"})
                return

            try:
                with model_lock:
                    result = direct_score(model, tokenizer, row, metadata)

                # Determine argmax decision
                probs = result["probabilities"]
                opt_ids = result["option_ids"]
                best_idx = max(range(len(probs)), key=lambda i: probs[i])
                best_option = opt_ids[best_idx]
                best_prob = probs[best_idx]

                decision_response = {
                    "id": row["id"],
                    "decision": best_option,
                    "confidence": best_prob,
                    "probabilities": dict(zip(opt_ids, probs)),
                    "option_logits": dict(zip(opt_ids, result["option_logits"])),
                    "input_tokens": result["input_tokens"],
                    "forward_seconds": result["forward_seconds"],
                    "total_seconds": result["total_seconds"],
                }
                self._send_json(HTTPStatus.OK, decision_response)
            except Exception as e:
                logger.exception("Error during scoring")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})

        elif self.path == "/v1/score":
            # Raw SemIf row scoring endpoint
            try:
                validate_row(payload)
                with model_lock:
                    result = direct_score(model, tokenizer, payload, metadata)
                self._send_json(HTTPStatus.OK, result)
            except Exception as e:
                logger.exception("Error during score")
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(e)})

        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint", "path": self.path})


def load_model_background(model_name: str, revision: str, device: str, dtype: str):
    """Load model in background so server can accept health check immediately."""
    global model, tokenizer, metadata, server_ready, model_load_error
    logger.info("Loading model: %s (revision: %s, device: %s, dtype: %s)...", model_name, revision, device, dtype)
    try:
        loaded_model, loaded_tok, meta = load_causal_model(
            source=model_name,
            revision=revision,
            device=device,
            dtype=dtype,
        )
        model = loaded_model
        tokenizer = loaded_tok
        metadata = meta
        server_ready = True
        logger.info("Model loaded successfully on %s (%s)!", meta["device"], meta["dtype"])
    except Exception as e:
        model_load_error = str(e)
        logger.exception("Failed to load model: %s", e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("SEMIF_HOST", DEFAULT_HOST), help="Host address")
    parser.add_argument("--port", type=int, default=int(os.getenv("SEMIF_PORT", DEFAULT_PORT)), help="Port number")
    parser.add_argument("--model", default=os.getenv("SEMIF_MODEL", DEFAULT_MODEL), help="Model source/path")
    parser.add_argument("--revision", default=os.getenv("SEMIF_REVISION", DEFAULT_REVISION), help="Model revision")
    parser.add_argument("--device", default=os.getenv("SEMIF_DEVICE", "mps"), choices=("mps", "cuda", "auto"), help="Device")
    parser.add_argument("--dtype", default=os.getenv("SEMIF_DTYPE", "float16"), choices=("float16", "bfloat16", "float32"), help="Precision")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SemIfRequestHandler)
    logger.info("Starting SemIf HTTP server on http://%s:%d", args.host, args.port)

    # Start model loading thread
    loader_thread = threading.Thread(
        target=load_model_background,
        args=(args.model, args.revision, args.device, args.dtype),
        daemon=True,
    )
    loader_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
