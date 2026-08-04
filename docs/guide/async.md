# Async & Streaming

`data-harness` ships `AsyncAgent` and `AsyncAgentSession` for async workflows
and token-level streaming. They mirror the synchronous API exactly, with
`await` on coroutines and `async for` on generators.

---

## AsyncAgent

```python
import asyncio
from data_harness import AsyncAgent
from data_harness.llm.providers.anthropic import AnthropicAdapter

async def main():
    agent = AsyncAgent(
        adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
        system="You are a data analyst.",
    )
    result = await agent.run("Compute the mean of [1, 2, 3, 4, 5].")
    print(result)

asyncio.run(main())
```

`AsyncAgent` requires an `AsyncProviderAdapter`. The built-in `AnthropicAdapter`
implements both `ProviderAdapter` and `AsyncProviderAdapter`.

---

## Streaming

Use `AsyncAgent.run_stream()` to receive token-level events as they arrive:

```python
async def stream_example():
    agent = AsyncAgent(
        adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
        system="You are a data analyst.",
    )

    async for event in agent.run_stream("Describe the unemployment trend."):
        match event.type:
            case "content_block_delta":
                from data_harness import TextDelta
                if isinstance(event.delta, TextDelta):
                    print(event.delta.text, end="", flush=True)
            case "tool_result":
                print(f"\n[tool: {event.tool_name}] {event.content[:80]}")
```

---

## Stream event types

The stream emits a sequence of typed events. Each event has a `type`
discriminator field matching the Anthropic SDK's raw SSE event shape:

| Event type | When |
|---|---|
| `message_start` | Before the first content block |
| `content_block_start` | A new text or tool-use block begins |
| `content_block_delta` | A text or JSON delta arrives |
| `content_block_stop` | The current block is complete |
| `message_delta` | Token counts and stop reason for this turn |
| `message_stop` | After the last content block |
| `tool_result` | After the harness dispatches a tool call |

`ToolResultEvent` is data-harness-specific; the raw provider stream does not
emit it.

---

## Async sessions

```python
async def session_example():
    agent = AsyncAgent(
        adapter=AnthropicAdapter(model="claude-sonnet-4-6"),
        system="You are a data analyst.",
    )
    session = agent.async_session()

    import pandas as pd
    session.put("sales", pd.read_csv("sales.csv"))

    print(await session.ask("What is the total revenue?"))
    print(await session.ask("Which product category was highest?"))
```

Streaming follow-up turns use `ask_stream()`:

```python
async for event in session.ask_stream("Summarise the analysis."):
    if event.type == "content_block_delta":
        from data_harness import TextDelta
        if isinstance(event.delta, TextDelta):
            print(event.delta.text, end="", flush=True)
```

---

## Implementing a streaming adapter

Override `stream_events()` in your `AsyncProviderAdapter` to emit real
token-level events. The default implementation calls `chat()` and synthesises
the standard event sequence from the assembled response:

```python
from data_harness.llm.providers.base import AsyncProviderAdapter, NormalizedResponse
from data_harness.llm.streaming import StreamEvent
from collections.abc import AsyncGenerator

class MyAdapter(AsyncProviderAdapter):
    async def chat(self, system, messages, tools) -> NormalizedResponse:
        ...

    def format_cache_control(self, obj):
        return obj

    async def stream_events(
        self, system, messages, tools
    ) -> AsyncGenerator[StreamEvent, None]:
        # Yield real token events from your provider SDK here
        ...
```
