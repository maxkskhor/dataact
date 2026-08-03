"""Facades: the API most users touch.

`Agent`/`AsyncAgent`, the zero-config `ask`/`Chat`/`SmartFrame` entry points,
the CLI, and notebook helpers. Composition only; no behaviour that belongs to
a lower layer.

May import everything below it.
"""
