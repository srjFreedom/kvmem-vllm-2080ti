"""Clean-room ring window for KVMem Plan D (no knife, no legacy state machine).

The whole protocol is three invariants, each one line:

  1. BOOK.  The scheduler slides `num_computed_tokens` by WHOLE PAGES
     (`KVMEM_SLIDE_CORE=1`), so `allocate_slots` and the engine block table never grow
     past the window.  The prompt feed is unaffected: the V2 runner slices prompt ids
     from the LOGICAL counter (`_prepare_prefill_inputs_kernel` uses `num_logical_tokens`),
     never from the resident book.

  2. ROW.  View column `c` must hold the engine page of absolute page
     `base // pt + c`, where `base = true_frontier - resident` and `pt` is the page
     token count.  While `base == 0` the engine's own row is already in this frame, so
     the hook only mirrors it.  Once `base > 0` the hook rotates the mirrored head by
     whole pages and installs it on the attention group; the pages that fall out are
     copied to the host store (absolute page keys) for later selection.

  3. PHASE.  Positions and slot addresses are the RESIDENT (view) frame, so the slot
     delta stays 0; RoPE must keep the TRUE position, so the hook publishes
     `kvmem_rope_offset = base`.  The attention length is the resident book
     (`kvmem_view_lens[row] + query_len`).

Everything else (knife, pressure, sched_compact, rebuild, session parking) is out of
scope here on purpose: the goal is a reproducible over-pool ring.
"""

from __future__ import annotations

import os
import traceback

import torch


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except Exception:
        return default


def _log(path: str, rec: dict) -> None:
    try:
        import json
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class RingClean:
    """Per-runner ring: head mirror, rotation, install, store."""

    def __init__(self, runner, log_path: str):
        self.runner = runner
        self.log = log_path
        self.pt = max(1, _env_int("KVMEM_PAGE_TOKENS", 32))
        win = _env_int("KVMEM_WINDOW_TOKENS", 0)
        self.win_pages = max(1, win // self.pt) if win > 0 else 0
        self.gi = self._find_attention_group()
        self.state: dict[str, dict] = {}
        self.store: dict[str, dict] = {}
        self.store_bytes = 0
        self.store_cap = _env_int("KVMEM_STORE_MIB", 8192) * 1024 * 1024
        self.probe_n = 0
        self.probe_max = _env_int("KVMEM_RING_PROBE_MAX", 400)

    # -- discovery ---------------------------------------------------------------
    def _find_attention_group(self) -> int:
        try:
            groups = getattr(self.runner.kv_cache_config, "kv_cache_groups", []) or []
            for i, g in enumerate(groups):
                if type(getattr(g, "kv_cache_spec", None)).__name__ == "FullAttentionSpec":
                    return i
        except Exception:
            pass
        return 0

    def _table(self):
        return self.runner.block_tables.block_tables[self.gi]

    def _row_np(self) -> list[int]:
        return self._table()[0].tolist()

    def _npages(self) -> int:
        try:
            return int(self.runner.block_tables.num_blocks.np[self.gi, 0])
        except Exception:
            return 0

    def _paged_kv(self):
        """Per-tensor KV caches, first dim = page id."""
        caches = getattr(self.runner, "kv_caches", None) or []
        out = []
        for t in caches:
            if t is None or not hasattr(t, "shape") or t.dim() < 2:
                continue
            try:
                if int(t.shape[0]) <= 1:
                    continue
            except Exception:
                continue
            out.append(t)
        return out

    # -- store -------------------------------------------------------------------
    def _commit(self, req: str, abs_page0: int, pages: list[int]) -> int:
        """Copy the evicted pages' bytes to the host store, keyed by absolute page."""
        if not pages:
            return 0
        paged = self._paged_kv()
        if not paged:
            return 0
        slot = self.store.setdefault(req, {})
        npages = self._npages()
        stored = 0
        with torch.no_grad():
            for k, page in enumerate(pages):
                key = int(abs_page0) + k
                p = int(page)
                if key in slot or p <= 0 or (npages and p >= npages):
                    continue
                if self.store_bytes >= self.store_cap:
                    break
                try:
                    blk = [t[p].detach().to("cpu", copy=True) for t in paged]
                except Exception as exc:
                    _log(self.log, {"kind": "ring_store_error", "req": req[-6:],
                                    "page": p, "error": str(exc)[:120]})
                    break
                slot[key] = blk
                self.store_bytes += sum(int(b.numel()) * int(b.element_size()) for b in blk)
                stored += 1
        return stored

    # -- the step ----------------------------------------------------------------
    def step(self, idx_mapping, query_start_loc, num_computed, positions, seq_lens) -> None:
        runner = self.runner
        qsl = query_start_loc.tolist() if hasattr(query_start_loc, "tolist") else list(query_start_loc)
        nreq = qsl.__len__() - 1
        if nreq <= 0:
            return
        req_ids = list(getattr(getattr(runner, "input_batch", None), "req_ids", None) or [])
        if len(req_ids) != nreq:
            return
        imap = idx_mapping.tolist() if hasattr(idx_mapping, "tolist") else list(idx_mapping)
        try:
            computed_np = runner.req_states.num_computed_tokens_np
        except Exception:
            computed_np = None
        try:
            logical_np = runner.req_states.num_logical_tokens_np
        except Exception:
            logical_np = None
        if computed_np is None or logical_np is None:
            return
        try:
            tbl = self._table()
            dv = getattr(runner, "kvmem_slot_deltas", None)
            vl = getattr(runner, "kvmem_view_lens", None)
            ro = getattr(runner, "kvmem_rope_offset", None)
            n_rows = int(getattr(runner, "max_num_reqs", tbl.shape[0]))
            if dv is None or vl is None:
                with torch.inference_mode(False):
                    dv = dv if dv is not None else torch.zeros(n_rows, dtype=torch.int32, device="cuda")
                    vl = vl if vl is not None else torch.zeros(n_rows, dtype=torch.int32, device="cuda")
                runner.kvmem_slot_deltas = dv
                runner.kvmem_view_lens = vl
            if ro is None:
                with torch.inference_mode(False):
                    ro = torch.zeros((), dtype=torch.int64, device="cuda")
                runner.kvmem_rope_offset = ro
            if getattr(runner, "kvmem_slot_offset", None) is None:
                with torch.inference_mode(False):
                    runner.kvmem_slot_offset = torch.zeros((), dtype=torch.int64, device="cuda")
        except Exception as exc:
            _log(self.log, {"kind": "ring_table_error", "error": str(exc)[:160]})
            return

        rope = 0
        for i, req in enumerate(req_ids):
            if str(req).startswith(("_warmup", "_v2_mixed")):
                continue
            rs = int(imap[i]) if i < len(imap) else i
            view = int(computed_np[rs])
            true = int(logical_np[rs])
            n_new = int(qsl[i + 1]) - int(qsl[i])
            base = true - view
            if base < 0:
                base = 0
            rec = {"kind": "ring_clean", "req": str(req)[-6:], "view": view,
                   "true": true, "n_new": n_new, "base": base, "pt": self.pt,
                   "win_pages": self.win_pages}
            st = self.state.get(req)
            if st is None:
                st = {"head": (), "roll": 0}
                self.state[req] = st
            if base % self.pt:
                rec["state"] = "unaligned"
                self._probe(rec)
                continue
            need_pages = max(1, (view + n_new + self.pt - 1) // self.pt)
            eng = self._row_np()
            npages = self._npages()
            width = self.win_pages or need_pages
            if base == 0:
                want = min(width, max(need_pages, 1))
                head = tuple(int(p) for p in eng[:want] if 0 < int(p) < (npages or (1 << 30)))
                st["head"] = head
                st["roll"] = 0
                rec.update(state="identity", head_distinct=len(set(head)),
                           rowlen=len(head), true_after=true + n_new,
                           pos_shift=0, pos_head=[int(x) for x in positions[int(qsl[i]):int(qsl[i]) + 2].tolist()])
                self._probe(rec)
                continue
            # extend the head from the engine row before rotating: the engine writes its
            # own (view-frame) page ids at the columns its appends reach.
            head = list(st["head"])
            want = min(width, need_pages)
            if len(head) < want:
                head.extend(int(p) for p in eng[len(head):want])
            head = [p for p in head[:width] if 0 < p < (npages or (1 << 30))]
            if not head:
                rec["state"] = "no_head"
                self._probe(rec)
                continue
            steps = (base - int(st["roll"])) // self.pt
            if steps < 0 or steps >= len(head):
                rec.update(state="bad_roll", steps=steps, roll=st["roll"], head=len(head))
                self._probe(rec)
                continue
            if steps:
                rec["stored"] = self._commit(req, int(st["roll"]) // self.pt, head[:steps])
                head = head[steps:] + head[:steps]
                st["roll"] = int(st["roll"]) + steps * self.pt
            st["head"] = tuple(head)
            row = head[:width]
            try:
                tbl[:][0, :len(row)] = torch.tensor(row, dtype=tbl.dtype, device=tbl.device)
                runner.block_tables.num_blocks.np[self.gi, 0] = len(row)
            except Exception as exc:
                rec.update(state="install_error", error=str(exc)[:120])
                self._probe(rec)
                continue
            dv[0] = 0
            vl[0] = view
            rope = base
            rec.update(state="slid", head_distinct=len(set(row)), rowlen=len(row),
                       row_head=[int(x) for x in row[:4]],
                       true_after=true + n_new, pos_head=[
                           int(x) for x in positions[int(qsl[i]):int(qsl[i]) + 2].tolist()])
            self._probe(rec)
        try:
            with torch.no_grad():
                ro.fill_(int(rope))
            runner.kvmem_view_active = True
        except Exception:
            pass

    def _probe(self, rec: dict) -> None:
        if self.probe_n >= self.probe_max:
            return
        self.probe_n += 1
        _log(self.log, rec)


def install(mr, log_path: str) -> None:
    """Patch `prepare_pos_seq_lens` with the ring hook (idempotent)."""
    original = mr.prepare_pos_seq_lens
    if getattr(original, "_kvmem_ring_clean", False):
        return
    state = {"runners": {}}

    def prepare_pos_seq_lens(self, idx_mapping, query_start_loc, num_computed_tokens,
                             positions, seq_lens):
        original(self, idx_mapping, query_start_loc, num_computed_tokens, positions, seq_lens)
        ring = state["runners"].get(id(self))
        try:
            if ring is None:
                ring = RingClean(self, log_path)
                state["runners"][id(self)] = ring
            ring.step(idx_mapping, query_start_loc, num_computed_tokens, positions, seq_lens)
        except Exception:
            _log(log_path, {"kind": "ring_clean_error", "error": traceback.format_exc()[-600:]})

    prepare_pos_seq_lens._kvmem_ring_clean = True
    mr.prepare_pos_seq_lens = prepare_pos_seq_lens
    _log(log_path, {"kind": "ring_clean_installed", "pid": os.getpid()})
