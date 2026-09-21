from __future__ import annotations

import uuid
from collections.abc import Callable

from starlette.types import ASGIApp


class RequestIdMiddleware:
    """Attach a request ID to every response for traceability."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode("utf-8").strip()
        if not request_id:
            request_id = str(uuid.uuid4())

        async def send_with_request_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("utf-8")))
                message = {**message, "headers": response_headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)
