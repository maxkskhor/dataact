"""The session tree: an append-only log of typed entries, shaped like a tree.

The conversation the model sees is *derived* from the tree by walking root to
leaf. It is never stored, so there is no second copy to disagree with the log.
Compaction is an entry that changes where the walk starts; retrying a turn is
a second child of the same parent; both branches survive.
"""

from data_harness.core.session.entries import (
    ENTRY_TYPES,
    BaseEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    TurnEntry,
    new_entry_id,
    utc_now,
)
from data_harness.core.session.jsonl import (
    FORMAT_VERSION,
    JsonlSessionStore,
    encode_entry,
)
from data_harness.core.session.session import (
    CustomProjector,
    Session,
    SessionStats,
    leaf_entries,
)
from data_harness.core.session.store import (
    MemorySessionStore,
    SessionStore,
    SessionStoreError,
)

__all__ = [
    "ENTRY_TYPES",
    "FORMAT_VERSION",
    "BaseEntry",
    "CompactionEntry",
    "CustomEntry",
    "CustomProjector",
    "Entry",
    "JsonlSessionStore",
    "LabelEntry",
    "LeafEntry",
    "MemorySessionStore",
    "MessageEntry",
    "Session",
    "SessionStats",
    "SessionStore",
    "SessionStoreError",
    "TurnEntry",
    "encode_entry",
    "leaf_entries",
    "new_entry_id",
    "utc_now",
]
