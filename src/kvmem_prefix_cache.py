"""Explicit prefix cache: the single, qualified checkpoint store for cross-request reuse.

Design rule (from the upstream kvmem-vllm prompt-sessions spec):

* identity is the **block-hash chain** of the prompt, never a client-declared session;
* a record may only be published for a prefix the request **executed in full** (a compacted
  view is a selection, so it can never describe "the first N tokens");
* a record is only usable when it carries the **recurrent state** captured at that frontier;
* the pages are stored under **logical** indices (page i == tokens [i*pt, (i+1)*pt)), so a
  restoring request writes logical page i into its own i-th page and nothing has to be
  re-derived from physical ids.

Everything the legacy tree got wrong is expressed as a refusal here: no state, incomplete
coverage, digest mismatch, compacted origin, or a publish from another request id all mean
"no record" and the caller recomputes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


class PrefixCacheError(RuntimeError):
    """A record failed one of the publish/accept invariants."""


def _digest_token_ids(ids, upto: int) -> str:
    h = hashlib.sha256()
    for token in list(ids)[: int(upto)]:
        h.update(int(token).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def block_hash_chain(ids, block_size: int):
    """Chain of per-block digests; identical chains mean an identical token prefix."""
    block_size = max(1, int(block_size))
    tokens = list(ids)
    chain = []
    prev = b""
    for start in range(0, (len(tokens) // block_size) * block_size, block_size):
        block = tokens[start:start + block_size]
        h = hashlib.sha256(prev)
        for token in block:
            h.update(int(token).to_bytes(4, "little", signed=False))
        digest = h.digest()
        chain.append(digest)
        prev = digest
    return tuple(chain)


@dataclass
class PrefixRecord:
    chain: tuple
    covered: int
    page_tokens: int
    digest: str
    pages: dict = field(default_factory=dict)   # logical index -> host tensors
    state: Any = None                            # GDN/StateImage at the frontier
    request: str = ""
    sealed_full_prefix: bool = False

    @property
    def has_state(self) -> bool:
        return self.state is not None

    @property
    def page_count(self) -> int:
        return int(self.covered) // max(1, int(self.page_tokens))

    def usable_pages(self):
        """Logical indices actually present, ascending. Must be 0..page_count-1 to be usable."""
        return sorted(int(k) for k in self.pages)


class PrefixCache:
    """Small LRU of qualified prefix records, keyed by the chain tail."""

    def __init__(self, capacity: int = 8):
        self.capacity = max(1, int(capacity))
        self._records: dict = {}
        self._order: list = []

    def __len__(self):
        return len(self._records)

    # --- publish -----------------------------------------------------------------
    def publish(self, *, token_ids, block_size, page_tokens, pages, state, request_id,
                compacted_in_prefill: bool):
        """Register a checkpoint, or refuse. Refusals are the point of this method."""
        if compacted_in_prefill:
            raise PrefixCacheError("compacted prefill cannot describe a sealed prefix")
        if state is None:
            raise PrefixCacheError("no recurrent state at the frontier")
        page_tokens = max(1, int(page_tokens))
        chain = block_hash_chain(token_ids, block_size)
        if not chain:
            raise PrefixCacheError("prompt shorter than one block")
        covered = len(chain) * max(1, int(block_size))
        expected = covered // page_tokens
        present = sorted(int(k) for k in (pages or {}))
        if present != list(range(expected)):
            raise PrefixCacheError(
                f"logical pages must be exactly 0..{expected - 1}, got {len(present)}")
        record = PrefixRecord(chain=chain, covered=covered, page_tokens=page_tokens,
                              digest=_digest_token_ids(token_ids, covered),
                              pages=dict(pages), state=state,
                              request=str(request_id), sealed_full_prefix=True)
        key = chain[-1]
        self._records[key] = record
        self._order = [k for k in self._order if k != key] + [key]
        while len(self._order) > self.capacity:
            self._records.pop(self._order.pop(0), None)
        return record

    # --- lookup ------------------------------------------------------------------
    def lookup(self, token_ids, block_size: int):
        """Return a record whose chain is a prefix of this prompt, or None."""
        chain = block_hash_chain(token_ids, block_size)
        if not chain:
            return None
        for tail in reversed(self._order):
            record = self._records.get(tail)
            if record is None:
                continue
            depth = len(record.chain)
            if depth > len(chain) or chain[:depth] != record.chain:
                continue
            if _digest_token_ids(token_ids, record.covered) != record.digest:
                continue
            if not record.has_state:
                continue
            if record.usable_pages() != list(range(record.page_count)):
                continue
            return record
        return None


# The record holds KV tensors in PROCESS memory, so the cache must live as long as the process,
# not as long as any one runtime object: a per-instance cache lost every record between the
# publish step and the fill step of the very next request (measured: "no record", cache_n=-1,
# while the same pid had just published 902 pages).
_SHARED: PrefixCache | None = None


def shared_cache(capacity: int = 8) -> PrefixCache:
    global _SHARED
    if _SHARED is None:
        _SHARED = PrefixCache(capacity=capacity)
    return _SHARED


@dataclass
class FrameRecord:
    """A compacted working set: the view pages in VIEW order plus the recurrent state.

    A prompt longer than the resident budget never executes as one prefix, so a plain
    prefix record cannot describe it. What DOES exist at the end of its prefill is the
    compacted frame: `pages[i]` is the KV of view slot i, `logical[i]` the logical page that
    slot represents, and `state` the recurrent state at that frontier. Restoring the frame
    (pages + order + state) reproduces the same context for an identical prompt.
    """

    chain: tuple
    covered: int          # prompt tokens that were executed
    view_tokens: int      # tokens the frame holds (view length)
    page_tokens: int
    digest: str
    pages: dict = field(default_factory=dict)   # view slot -> host tensors
    logical: tuple = ()
    state: Any = None
    request: str = ""

    @property
    def has_state(self) -> bool:
        return self.state is not None

    @property
    def page_count(self) -> int:
        return int(self.view_tokens) // max(1, int(self.page_tokens))


class FrameCache:
    """Small LRU of compacted frames, keyed by the prompt chain tail."""

    def __init__(self, capacity: int = 2):
        self.capacity = max(1, int(capacity))
        self._records: dict = {}
        self._order: list = []

    def __len__(self):
        return len(self._records)

    def publish(self, *, token_ids, block_size, page_tokens, covered, view_tokens,
                pages, logical, state, request_id):
        if state is None:
            raise PrefixCacheError("frame without recurrent state")
        if int(covered) <= 0 or int(view_tokens) <= 0:
            raise PrefixCacheError("frame needs covered>0 and view_tokens>0")
        page_tokens = max(1, int(page_tokens))
        expected = int(view_tokens) // page_tokens
        present = sorted(int(k) for k in (pages or {}))
        if present != list(range(expected)):
            raise PrefixCacheError(
                f"frame slots must be exactly 0..{expected - 1}, got {len(present)}")
        if len(tuple(logical)) < expected:
            raise PrefixCacheError("frame logical map shorter than the slot count")
        chain = block_hash_chain(token_ids, block_size)
        if not chain:
            raise PrefixCacheError("prompt shorter than one block")
        record = FrameRecord(chain=chain, covered=int(covered), view_tokens=int(view_tokens),
                             page_tokens=page_tokens,
                             digest=_digest_token_ids(token_ids, int(covered)),
                             pages=dict(pages), logical=tuple(int(x) for x in
                                                              tuple(logical)[:expected]),
                             state=state, request=str(request_id))
        key = chain[-1]
        self._records[key] = record
        self._order = [k for k in self._order if k != key] + [key]
        while len(self._order) > self.capacity:
            self._records.pop(self._order.pop(0), None)
        return record

    def lookup(self, token_ids, block_size: int):
        chain = block_hash_chain(token_ids, block_size)
        if not chain:
            return None
        for tail in reversed(self._order):
            record = self._records.get(tail)
            if record is None:
                continue
            depth = len(record.chain)
            if depth > len(chain) or chain[:depth] != record.chain:
                continue
            if _digest_token_ids(token_ids, record.covered) != record.digest:
                continue
            if not record.has_state:
                continue
            if sorted(int(k) for k in record.pages) != list(range(record.page_count)):
                continue
            return record
        return None


_SHARED_FRAMES: FrameCache | None = None


def shared_frames(capacity: int = 2) -> FrameCache:
    global _SHARED_FRAMES
    if _SHARED_FRAMES is None:
        _SHARED_FRAMES = FrameCache(capacity=capacity)
    return _SHARED_FRAMES
