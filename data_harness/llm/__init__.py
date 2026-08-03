"""Provider layer: the wire format and everything that speaks it.

Owns the types a model provider exchanges (`Message`, `ToolSpec`,
`ContentBlock`), the streaming event protocol, and the adapters that translate
provider SDKs into them.

Knows nothing about agents, tools-as-behaviour, caches, or DataFrames. It is
the bottom of the stack: nothing here may import `core`, `data`, or `app`.
"""
