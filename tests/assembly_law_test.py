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
    share = max(0, room - len(mid))   # floor at 0 (r98 assertion find)
    seq = list(order)[:share] + list(mid)
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
    order2 = [45, 46, 89, 90]
    mid = list(range(80, 95))
    k2 = assemble(18, {0}, {91}, order2, mid)
    for b in order2[:2]:
        if b not in k2:
            print(f"FAIL: top-k rank{order2.index(b)} block {b} crowded out by mid"); fails += 1
    # rank beyond room is NOT guaranteed (documented ceiling behaviour)
    order3 = list(range(100, 109)) + [777]
    k3 = assemble(12, {0, 1}, {2, 3}, order3, [])
    if 777 in k3:
        print("NOTE: deep-rank block admitted anyway (fine, not a violation)")
    if fails == 0:
        print("assembly law assertions: ALL PASS")
    return 1 if fails else 0


sys.exit(main())
