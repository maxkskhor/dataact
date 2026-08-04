# SessionCache

In-process object store that exposes large values as named handles with compact
snapshots. The model never sees raw data in message history; it operates on
handles through the Python interpreter.

---

::: data_harness.data.cache.SessionCache
