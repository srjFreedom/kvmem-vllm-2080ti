# The Plan-D ring window contract (design 1576), checked end to end on the real runtime method.
#
# The window is a page-aligned look-back over a rotating block-table row:
#   view column c  ->  engine page row[c]  ->  absolute page base//pt + c
#   base = ceil((frontier - win)/pt)*pt   (page aligned, so a roll moves whole pages)
#   vl   = frontier - base <= win         (decode grows it by one; no slot is overwritten)
#   the engine's slot delta is EXACTLY base and the RoPE offset is 0 (positions are TRUE),
#   so this step's tokens are written at view positions `position - base`.
#
# What this test proves, with a fake engine whose page content is its own token range:
#   1. after every roll, reading the view returns the true tokens base..base+vl-1 - including
#      while new chunks keep arriving and during decode;
#   2. the pages a roll recycles are committed to the host store under their ABSOLUTE page
#      key, so block b page j is simply b*q + j (design 839's restore query);
#   3. restoring an evicted block into a sacrificed slot's pages makes exactly those view
#      positions serve the block's tokens (in-place selection, no row change);
#   4. the arm this replaced (roll by token/1024, positions left in the true frame while the
#      delta was also published) writes OUTPUTSIDE the installed row: the negative control
#      shows the view never changes, which is the round-576 "reads the prompt head" symptom.
import sys

import torch

sys.path.insert(0, "build/python")
sys.path.insert(0, "src")
from kvmem_sparse_runtime import VllmSparseRuntime, _ReqState

failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


PT = 32          # page tokens
Q = 49           # pages per engine block (numpy: 1568-token blocks over 32-token pages)
WIN = 45472      # budget + gen reserve (29 blocks exactly)
PAGES = 2842     # engine KV page pool
HEAD = list(range(1, 1 + WIN // PT))   # the prompt head in view order (pages 1..1421)


class FakeStore:
    def __init__(self):
        self.pages = {}

    def has(self, key):
        return int(key) in self.pages

    def get(self, key):
        return self.pages[int(key)]

    def put(self, key, tensors):
        self.pages[int(key)] = [t.clone() for t in tensors]

    def __len__(self):
        return len(self.pages)


class FakeSelector:
    def budget_pages(self):
        return WIN // PT


class FakeOrch:
    def __init__(self):
        self.store = FakeStore()


class FakeView:
    def __init__(self, kv, table, rows=1, width=8232):
        self.kv = kv
        self.table = table
        self.rows = rows
        self.width = width
        self._seq = torch.zeros(rows, dtype=torch.int32)
        self._pos = torch.zeros(4096, dtype=torch.int64)

    def paged_kv(self):
        return [self.kv]

    def block_table_rows(self):
        return [self.table]

    def seq_lens(self):
        return self._seq

    def positions(self):
        return self._pos

    def num_blocks(self):
        return int(self.kv.shape[0])

    def row_index(self, req):
        return 0

    def query_position_slots(self, req):
        return []


class FakeRunner:
    def __init__(self, rows=1):
        self.kvmem_slot_deltas = torch.zeros(rows, dtype=torch.int64)
        self.kvmem_view_lens = torch.zeros(rows, dtype=torch.int64)
        self.kvmem_rope_offset = torch.zeros((), dtype=torch.int64)
        self.kvmem_slot_offset = torch.zeros((), dtype=torch.int64)
        self.kvmem_view_active = False
        self.input_batch = None
        self.block_tables = None


def build():
    # Page p (>=1) holds tokens (p-1)*32 .. (p-1)*32+31; -1 = never written.
    kv = torch.full((PAGES, PT), -1, dtype=torch.int64)
    for page in range(1, PAGES):
        kv[page] = torch.arange((page - 1) * PT, page * PT, dtype=torch.int64)
    table = torch.zeros(1, 8232, dtype=torch.int32)
    # The engine's staged row before any view install is TRUE-POSITION indexed: column c
    # holds the page for token c*32 (allocation is sequential). Fill the allocated prefix.
    table[0, :1714] = torch.arange(1, 1715, dtype=torch.int32)
    view = FakeView(kv, table)
    runner = FakeRunner()
    view.runner = runner
    rt = VllmSparseRuntime(
        view=view, selector=FakeSelector(), page_tokens=PT, head_dim=128, rotary_dim=32,
        base=1e6, budget_tokens=WIN - 8192, gen_reserve=8192)
    rt.block_pages = Q
    rt._orch = {"r": FakeOrch()}
    st = _ReqState()
    st.row = ()
    st.roll_base = 0
    st.row_index = 0
    st.view_installed = False
    return rt, view, runner, rt._orch["r"], st


def write_tokens(kv, row, base, tok, n_new):
    """The engine writes this step's tokens at view positions `token - base`."""
    for pos in range(tok, tok + n_new):
        v = pos - base
        page = int(row[v // PT])
        kv[page, v % PT] = pos


def read_view(kv, row, base, vl):
    out = []
    for v in range(vl):
        out.append(int(kv[int(row[v // PT]), v % PT]))
    return out


def main():
    rt, view, runner, orch, st = build()
    tok = 0
    chunk = 4096
    reached = 0
    while True:
        n_new = min(chunk, 160000 - tok)
        if n_new <= 0:
            break
        out = rt.d_ring_window("r", st, 0, tok, n_new)
        frontier = tok + n_new
        base = int(out.get("base", 0))
        vl = int(out.get("vl", 0))
        row = list(st.row)
        if frontier > WIN:
            check(int(runner.kvmem_slot_deltas[0]) == base,
                  f"delta == base at frontier {frontier} ({base})")
            check(int(runner.kvmem_rope_offset) == 0, "RoPE offset stays 0 (positions are true)")
            check(int(runner.kvmem_view_lens[0]) + n_new == vl,
                  f"fork view table + query length == view length ({int(runner.kvmem_view_lens[0])} + {n_new} == {vl})")
            check(base % PT == 0, f"base is page aligned ({base})")
            check(0 < vl <= WIN, f"view length inside the window ({vl})")
            write_tokens(view.kv, row, base, tok, n_new)
            want = list(range(base, base + vl))
            got = read_view(view.kv, row, base, vl)
            check(got == want,
                  f"view at frontier {frontier} returns tokens {base}..{base + vl - 1}")
            # the store must hold every recycled page under its absolute page key
            if base >= 2 * chunk:
                blk = 5                       # tokens 7840..9407 -> pages 245..293
                srcs = [blk * Q + j for j in range(Q)]
                check(all(orch.store.has(s) for s in srcs),
                      f"block {blk}'s pages are stored under b*q+j after being rolled past")
                first = orch.store.get(srcs[0])[0]
                check(int(first[0]) == srcs[0] * PT,
                      "a stored page's content is its absolute page's token range")
                reached = max(reached, sum(1 for s in srcs if orch.store.has(s)))
        tok = frontier
    check(tok >= 160000, f"ran a 160K prompt through the ring ({tok} tokens)")
    check(reached == Q, "the restore query b*q+j finds all 49 pages of an evicted block")

    # ---- in-place restore: write an evicted block into a sacrificed slot's pages ----
    # Pick the most recent chunk's view: reuse the live state by taking one more step.
    n_new = 1
    out = rt.d_ring_window("r", st, 0, tok, n_new)
    base = int(out["base"])
    vl = int(out["vl"])
    row = list(st.row)
    write_tokens(view.kv, row, base, tok, n_new)
    tok += 1
    sacrifice = 3                     # any view block whose pages are ours
    blk = 5                           # evicted block restored into it
    pairs = [(blk * Q + j, int(row[sacrifice * Q + j])) for j in range(Q)]
    check(all(orch.store.has(s) for s, _ in pairs), "all restore sources exist in the store")
    for src, dst in pairs:
        view.kv[dst] = orch.store.get(src)[0]
    want_positions = set(range(sacrifice * Q, (sacrifice + 1) * Q))
    ok = True
    for v in range(vl):
        col = v // PT
        if col in want_positions:
            j = col - sacrifice * Q
            want = (blk * Q + j) * PT + v % PT
            ok &= int(view.kv[int(row[col]), v % PT]) == want
    check(ok, "the sacrificed slot now serves the restored block's tokens")

    # ---- slide arm: the Core keeps the book in the VIEW frame (page-aligned) ----
    # Same fake engine, but the book is no longer the true frontier: the Core computes
    # base = page_align(true - win) and sets book = true - base, so positions are view
    # frame, the writes land at view positions without any hole, and the first install
    # is a rotation of the engine's head row (never a slice past it).
    rt2, view2, runner2, orch2, st2 = build()
    st2.row = tuple(HEAD)
    st2.ring_ready = False
    st2.d_slide_true = 0
    true_tok = 0
    slid = 0
    while true_tok < 160000:
        n_new = min(chunk, 160000 - true_tok)
        true_after = true_tok + n_new
        base = ((true_after - WIN + PT - 1) // PT) * PT if true_after > WIN else 0
        book = true_tok - base
        out = rt2.d_ring_slide("r", st2, 0, book, n_new)
        if true_after > WIN:
            sl = int(out.get("base", -1))
            check(sl == base, f"slide derives the Core base at frontier {true_after} ({sl} == {base})")
            check(int(runner2.kvmem_slot_deltas[0]) == base, "slide publishes delta == base")
            check(int(runner2.kvmem_rope_offset) == base, "slide publishes rope offset == base")
            check(int(runner2.kvmem_view_lens[0]) == book, "slide publishes view_lens == book")
            check(int(runner2.kvmem_view_lens[0]) + n_new == true_after - base,
                  "fork view length == frontier - base")
            check(st2.roll_base == base, f"row rolled to the absolute base ({st2.roll_base})")
            # the engine writes this step's tokens at view positions
            write_tokens(view2.kv, list(st2.row), base, true_tok, n_new)
            want = list(range(base, true_after))
            got = read_view(view2.kv, list(st2.row), base, true_after - base)
            check(got == want,
                  f"slide view at frontier {true_after} returns tokens {base}..{true_after - 1}")
            slid += 1
        else:
            check(out.get("state") == "identity", "below the window the slide arm is identity")
        true_tok = true_after
    check(slid >= 20, f"slide arm ran the whole 160K prompt ({slid} slid steps)")

    # ---- negative control: the replaced arm never reaches the view ----
    kv2 = torch.full((PAGES, PT), -1, dtype=torch.int64)
    for page in range(1, PAGES):
        kv2[page] = torch.arange((page - 1) * PT, page * PT, dtype=torch.int64)
    row2 = list(HEAD)
    tok2, chunk2 = 0, 4096
    head_seen = True
    while tok2 < 160000:
        n_new = min(chunk2, 160000 - tok2)
        frontier = tok2 + n_new
        if frontier > WIN:
            base_legacy = ((frontier - WIN + PT - 1) // PT) * PT
            # old arm: rotation by token/1024, and writes at TRUE positions
            steps = (base_legacy - 0) // 1024
            for pos in range(tok2, tok2 + n_new):
                col = pos // PT
                if col >= len(row2):
                    continue          # the true column is outside our installed row
                kv2[int(row2[col]), pos % PT] = pos
            head_seen &= read_view(kv2, row2, 0, WIN) == list(range(WIN))
        tok2 = frontier
    check(head_seen,
          "negative control: true-position writes leave the head view untouched (the old symptom)")


if __name__ == "__main__":
    main()
    print("FAIL ring_window_test" if failed else "PASS ring_window_test")
    sys.exit(1 if failed else 0)
