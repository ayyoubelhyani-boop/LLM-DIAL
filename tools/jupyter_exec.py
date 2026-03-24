from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid

import requests
import websocket


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Python code in a JupyterHub user kernel.")
    parser.add_argument("--base-url", required=True, help="JupyterHub base URL, for example http://host")
    parser.add_argument("--username", required=True, help="Jupyter username")
    parser.add_argument("--password", required=True, help="Jupyter password")
    parser.add_argument("--timeout", type=float, default=600.0, help="Websocket receive timeout in seconds")
    args = parser.parse_args()

    code = sys.stdin.read()
    if not code.strip():
        raise SystemExit("No code provided on stdin")

    output = execute_code(args.base_url, args.username, args.password, code, timeout=args.timeout)
    sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))


def execute_code(base_url: str, username: str, password: str, code: str, *, timeout: float) -> str:
    session = requests.Session()
    login_page = session.get(f"{base_url}/hub/login", timeout=30)
    login_page.raise_for_status()
    xsrf = re.search(r'name="_xsrf" value="([^"]+)"', login_page.text)
    if not xsrf:
        raise RuntimeError("Could not extract _xsrf token from login page")

    login_response = session.post(
        f"{base_url}/hub/login?next=%2Fhub%2F",
        data={"_xsrf": xsrf.group(1), "username": username, "password": password},
        allow_redirects=True,
        timeout=60,
    )
    login_response.raise_for_status()

    token = _extract_server_token(session, base_url, username)

    headers = {"Authorization": f"token {token}"}
    kernel_response = session.post(
        f"{base_url}/user/{username}/api/kernels",
        headers=headers,
        json={"name": "python3"},
        timeout=60,
    )
    kernel_response.raise_for_status()
    kernel_id = kernel_response.json()["id"]

    ws_base = re.sub(r"^http", "ws", base_url, count=1)
    cookie_header = "; ".join(f"{cookie.name}={cookie.value}" for cookie in session.cookies)
    ws = websocket.create_connection(
        f"{ws_base}/user/{username}/api/kernels/{kernel_id}/channels?session_id={uuid.uuid4()}&token={token}",
        header=[f"Cookie: {cookie_header}"],
    )
    ws.settimeout(timeout)

    try:
        _drain_startup_messages(ws)

        msg_id = uuid.uuid4().hex
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": username,
                "session": uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }
        ws.send(json.dumps(execute_request))

        chunks: list[str] = []
        while True:
            raw = ws.recv()
            message = json.loads(raw)
            if message.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = message.get("header", {}).get("msg_type") or message.get("msg_type")
            content = message.get("content", {})
            if msg_type == "stream":
                chunks.append(content.get("text", ""))
            elif msg_type == "execute_result":
                chunks.append(content.get("data", {}).get("text/plain", ""))
            elif msg_type == "error":
                chunks.append("ERROR\n" + "\n".join(content.get("traceback", [])))
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return "".join(chunks)
    finally:
        try:
            ws.close()
        finally:
            session.delete(f"{base_url}/user/{username}/api/kernels/{kernel_id}", headers=headers, timeout=30)


def _drain_startup_messages(ws: websocket.WebSocket, *, max_wait_seconds: float = 2.0) -> None:
    start = time.time()
    while time.time() - start < max_wait_seconds:
        try:
            raw = ws.recv()
            message = json.loads(raw)
            if (
                message.get("header", {}).get("msg_type") == "status"
                and message.get("content", {}).get("execution_state") == "idle"
            ):
                break
        except Exception:
            break


def _extract_server_token(session: requests.Session, base_url: str, username: str) -> str:
    pages = [
        f"{base_url}/user/{username}/tree",
        f"{base_url}/user/{username}/lab",
    ]
    for _ in range(3):
        for page in pages:
            response = session.get(page, timeout=60)
            response.raise_for_status()
            token_match = re.search(r'"token":\s*"([^"]+)"', response.text)
            if token_match:
                return token_match.group(1)
        time.sleep(1)
    raise RuntimeError("Could not extract server token from Jupyter user pages")


if __name__ == "__main__":
    main()
