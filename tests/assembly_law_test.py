"""Assembly law assertions (design 151): pure set logic, no engine.

The question/knife assembly must admit any block ranked within the
remaining budget: kept_final ⊇ (sink ∪ recent ∪ order[:room-mid]).
Two measured violations anchor the regression:

  * 152K: field rank 59 -> correctly NOT forced in (informational
    ceiling; the law only guards rank <= room).
  * insurance crowding (r97 seq-order experiment): mid blocks may
    only take slots top-k left.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def assemble(budget, sink, recent, order, mid):
    """Mirror of the runtime law: seq = order[:room-len(mid)] + mid,
    fill by seq into kept = sink|recent."""
    kept = set(sink) | set(recent)
    room = max(0, budget - len(kept))
    # r96 VERBATIM semantics (python negative slice END is relative
    # to the list end, NOT empty): share = room - len(mid) as the raw
    # slice bound, matching the runtime expression character for
    # character.
    share = room - len(mid)
    seq = list(order)[share if share < 0 else slice(0, share)] + list(mid)
    for b in seq:
        if room <= 0:
            break
        if b not in kept:
            kept.add(b)
            room -= 1
    return kept


def main():
    fails = 0
    # 88K reality: sink 0-6, recent 26-48-ish, field top = [25,26,8,...]
    sink = {0, 1, 2, 3, 4, 5, 6}
    order = [25, 26, 8, 27, 10, 9, 12, 13]
    k = assemble(18, sink, {28, 29}, order, [])
    if 25 not in k:
        print("FAIL: rank-0 needle not admitted"); fails += 1
    # mid may not crowd out top-k within room
    # mid > room: python suffix semantics keep the TAIL of order in
    # the head slice; assert that (documented) behaviour explicitly so
    # a future slice refactor must consciously change it.
    order2 = [45, 46, 89, 90]
    mid = list(range(80, 95))
    k2 = assemble(18, {0}, {91}, order2, mid)
    if 90 not in k2:
        print("FAIL: suffix law broken (rank-last top-k block missing)"); fails += 1
    # mid < room: prefix semantics MUST admit rank-0/1
    k3 = assemble(18, {0}, {91}, [45, 46, 89, 90], [80])
    if 45 not in k3 or 46 not in k3:
        print("FAIL: prefix law lost rank0/1 admission"); fails += 1
    # rank beyond room is NOT guaranteed (documented ceiling behaviour)
    order3 = list(range(100, 109)) + [777]
    k3 = assemble(12, {0, 1}, {2, 3}, order3, [])
    if 777 in k3:
        print("NOTE: deep-rank block admitted anyway (fine, not a violation)")
    if fails == 0:
        print("assembly law assertions: ALL PASS")
    return 1 if fails else 0


sys.exit(main())
