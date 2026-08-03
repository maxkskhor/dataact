"""data-harness: a constrained agent harness for data analysis.

The package is layered, bottom up:

- `data_harness.llm` — provider adapters and the wire types they speak
- `data_harness.core` — the ReAct loop, `RunResult`, run logging
- `data_harness.data` — the session cache, interpreter, SQL, connectors
- `data_harness.app` — `Agent`, `ask`, the CLI

Each layer may import the ones below it and no others; `tests/test_layers.py`
enforces that. The names below are the stable public surface and are unaffected
by where a class physically lives.
"""

from data_harness._legacy_paths import install as _install_legacy_paths

_install_legacy_paths()

from data_harness.app.agent import (  # noqa: E402
    Agent,
    AgentSession,
    AsyncAgent,
    AsyncAgentSession,
)
from data_harness.app.quickstart import (  # noqa: E402
    Chat,
    SmartFrame,
    ask,
    resolve_adapter,
    resolve_async_adapter,
)
from data_harness.core.artifacts import ChartArtifact  # noqa: E402
from data_harness.core.exceptions import (  # noqa: E402
    MaxTurnsExceeded,
    SubagentRecursionError,
    ToolNotFoundError,
)
from data_harness.core.result import CacheStorageInfo, RunResult, Usage  # noqa: E402
from data_harness.data.exec_cache import ExecutionCache  # noqa: E402
from data_harness.data.harness import AsyncHarness  # noqa: E402
from data_harness.data.io import load_dataframe  # noqa: E402
from data_harness.data.mcp import MCPClient, MCPServer, mcp_tool_specs  # noqa: E402
from data_harness.llm.providers.base import (  # noqa: E402
    AsyncProviderAdapter,
    NormalizedResponse,
    ProviderAdapter,
    StopReason,
)
from data_harness.llm.streaming import (  # noqa: E402
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ContentDelta,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    StreamEvent,
    TextDelta,
    ToolResultEvent,
)
from data_harness.llm.types import (  # noqa: E402
    ContentBlock,
    Message,
    TextBlock,
    ToolAnnotations,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

__all__ = [
    "Agent",
    "AgentSession",
    "AsyncAgent",
    "AsyncAgentSession",
    "AsyncHarness",
    "AsyncProviderAdapter",
    "CacheStorageInfo",
    "Chat",
    "ChartArtifact",
    "ContentBlock",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "ContentDelta",
    "ExecutionCache",
    "InputJSONDelta",
    "MCPClient",
    "MCPServer",
    "MaxTurnsExceeded",
    "Message",
    "MessageDeltaEvent",
    "MessageStartEvent",
    "MessageStopEvent",
    "NormalizedResponse",
    "ProviderAdapter",
    "RunResult",
    "SmartFrame",
    "StopReason",
    "StreamEvent",
    "SubagentRecursionError",
    "TextBlock",
    "TextDelta",
    "ToolAnnotations",
    "ToolNotFoundError",
    "ToolResultBlock",
    "ToolResultEvent",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "ask",
    "load_dataframe",
    "mcp_tool_specs",
    "resolve_adapter",
    "resolve_async_adapter",
]
