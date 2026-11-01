"""MissingAgentGuardMiddleware — makes a schema-only assembly fail loudly if it is ever executed.

Implemented on ``wrap_model_call`` on purpose: ``before_agent`` would add a graph node (and a
``branch:to:`` channel), and the schema-only assembly must stay node-for-node identical to the
real agents so the runtime can decode their checkpoints with it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

ModelHandler = Callable[[ModelRequest], ModelResponse]
AsyncModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]


class MissingAgentGuardMiddleware(AgentMiddleware):
    """Raise on the first model call. Adds no node to the graph."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self._reason = reason

    def wrap_model_call(self, request: ModelRequest, handler: ModelHandler) -> ModelResponse:
        """Never calls the handler."""
        raise RuntimeError(self._reason)

    async def awrap_model_call(self, request: ModelRequest, handler: AsyncModelHandler) -> ModelResponse:
        """Async twin."""
        raise RuntimeError(self._reason)
