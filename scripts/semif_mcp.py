#!/usr/bin/env python3
"""SemIf Model Context Protocol (MCP) Server for Claude Desktop & Claude Code.

Connects Claude Desktop / Claude Code to the resident SemIf HTTP server via stdio MCP.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = os.getenv("SEMIF_URL", "http://127.0.0.1:8765")

TOOL_DEFINITION = {
    "name": "semif_decide",
    "description": (
        "Make an instantaneous semantic decision/classification over unstructured context (state) "
        "using the local SemIf model running on Apple Silicon GPU (Metal/MPS). "
        "Evaluates options in a single forward pass (<100ms) reading native option logits without slow text decoding."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "Unstructured evidence, text, logs, or conversation context to evaluate.",
            },
            "question": {
                "type": "string",
                "description": "The decision question (e.g. 'Is there evidence that the deployment succeeded?').",
            },
            "options": {
                "type": "array",
                "description": "List of choices. Each can be a string or an object with 'id' and 'description'.",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["id"],
                        },
                    ]
                },
            },
        },
        "required": ["state", "question", "options"],
    },
}


def call_semif(payload: dict) -> dict:
    url = f"{DEFAULT_URL}/v1/decide"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to SemIf server at {DEFAULT_URL}. "
            f"Please ensure the SemIf background service is running (launchctl list | grep semif). Error: {e}"
        ) from e


def handle_request(req: dict) -> dict | None:
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "semif-mcp",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [TOOL_DEFINITION],
            },
        }

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "semif_decide":
            try:
                res = call_semif(arguments)
                text_content = json.dumps(res, ensure_ascii=False, indent=2)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": text_content}
                        ]
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [
                            {"type": "text", "text": f"Error calling SemIf: {e}"}
                        ]
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error handling message: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
