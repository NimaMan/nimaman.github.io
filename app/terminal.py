"""
Terminal WebSocket endpoint for nimamanafcom.

Spawns a PTY with /bin/bash and bridges it to WebSocket.
Requires valid session cookie for authentication.

Algorithm:
1. Accept WebSocket connection
2. Validate session cookie from request
3. If invalid: close with 4001 (Unauthorized)
4. If valid:
   a. Spawn /bin/bash in PTY
   b. Start async tasks for:
      - Reading PTY output -> send to WebSocket
      - Reading WebSocket input -> write to PTY
   c. Handle resize messages (JSON with type: "resize")
5. On disconnect:
   a. Terminate PTY process
   b. Clean up tasks

Security:
- Session validation happens before PTY spawn
- PTY runs as the current user (not root)
- Only accessible from Tailscale network
"""

import asyncio
import json
import os
import select
import signal

from fastapi import WebSocket, WebSocketDisconnect
from ptyprocess import PtyProcess

from app.auth import SESSION_COOKIE_NAME, validate_session_token


async def validate_websocket_session(websocket: WebSocket) -> bool:
    """Validate session cookie from WebSocket request."""
    session_token = websocket.cookies.get(SESSION_COOKIE_NAME)
    return validate_session_token(session_token)


async def terminal_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for terminal access.

    Spawns a PTY with bash and bridges to WebSocket.
    Validates session before accepting connection.
    """
    # Validate session before accepting
    if not await validate_websocket_session(websocket):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Spawn PTY with bash
    pty = PtyProcess.spawn(
        ["/bin/bash"],
        env={
            **os.environ,
            "TERM": "xterm-256color",
            "SHELL": "/bin/bash",
        },
        dimensions=(24, 80),  # Default size, will be resized
    )

    async def read_pty():
        """Read from PTY and send to WebSocket."""
        loop = asyncio.get_event_loop()

        def read_with_timeout():
            """Non-blocking read using select. Returns bytes, None if closed, empty if no data."""
            readable, _, _ = select.select([pty.fd], [], [], 0.1)
            if not readable:
                return b""
            try:
                return os.read(pty.fd, 4096)
            except OSError:
                return None

        try:
            while pty.isalive():
                data = await loop.run_in_executor(None, read_with_timeout)
                if data is None:
                    break
                if data:
                    await websocket.send_text(data.decode("utf-8", errors="replace"))
                else:
                    await asyncio.sleep(0.01)
        except (OSError, EOFError, Exception):
            pass

    async def write_pty():
        """Read from WebSocket and write to PTY."""
        try:
            while True:
                data = await websocket.receive_text()

                # Check for resize message (JSON starting with {)
                if data.startswith("{"):
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "resize":
                            pty.setwinsize(msg.get("rows", 24), msg.get("cols", 80))
                            continue
                    except json.JSONDecodeError:
                        pass

                pty.write(data.encode("utf-8"))
        except (WebSocketDisconnect, Exception):
            pass

    # Run both tasks concurrently
    read_task = asyncio.create_task(read_pty())
    write_task = asyncio.create_task(write_pty())

    try:
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [read_task, write_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    finally:
        # Clean up PTY
        if pty.isalive():
            try:
                pty.kill(signal.SIGTERM)
                # Give it a moment to terminate
                await asyncio.sleep(0.1)
                if pty.isalive():
                    pty.kill(signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        try:
            await websocket.close()
        except Exception:
            pass
