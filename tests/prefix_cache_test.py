"""Offline contract test for the explicit prefix cache.

The cache is the only place that may say "this prefix can be reused", so its refusals are the
contract: a compacted origin, a missing recurrent state, incomplete logical pages and a changed
prompt must all end in "no record" rather than in a partial restore.
"""
import sys

sys.path.insert(0, "src")
from kvmem_prefix_cache import (  # noqa: E402
    PrefixCache, PrefixCacheError, block_hash_chain)

BLOCK = 8
PAGE = 4
failed = False


def check(name, condition, detail=""):
    global failed
    if condition:
        print(f"  ok   {name}")
    else:
        failed = True
        print(f"  FAIL {name} {detail}")


def make_pages(covered, page_tokens=PAGE, origin=0):
    return {i: [f"p{origin}-{i}-{j}" for j in range(page_tokens)]
            for i in range(covered // page_tokens)}


def main():
    ids = list(range(64))
    cache = PrefixCache(capacity=2)

    print("publish refusals")
    for name, kwargs in (
        ("compacted origin", dict(compacted_in_prefill=True, state=object())),
        ("no state", dict(compacted_in_prefill=False, state=None)),
    ):
        try:
            cache.publish(token_ids=ids, block_size=BLOCK, page_tokens=PAGE,
                          pages=make_pages(64), request_id="r1", **kwargs)
            check(name, False, "publish was accepted")
        except PrefixCacheError as exc:
            check(name, True, str(exc))

    try:
        cache.publish(token_ids=ids, block_size=BLOCK, page_tokens=PAGE,
                      pages={i: ["x"] for i in range(3)}, request_id="r1",
                      state=object(), compacted_in_prefill=False)
        check("incomplete logical pages", False, "publish was accepted")
    except PrefixCacheError:
        check("incomplete logical pages", True)

    print("publish + lookup")
    record = cache.publish(token_ids=ids, block_size=BLOCK, page_tokens=PAGE,
                           pages=make_pages(64), request_id="r1", state=object(),
                           compacted_in_prefill=False)
    check("covered is block aligned", record.covered == 64, record.covered)
    hit = cache.lookup(ids, BLOCK)
    check("same prompt hits", hit is record)
    longer = ids + [999, 998, 1, 2, 3, 4, 5, 6]
    check("longer prompt hits the same prefix", cache.lookup(longer, BLOCK) is record)
    changed = list(ids)
    changed[40] ^= 0xFF
    check("changed prefix misses", cache.lookup(changed, BLOCK) is None)
    check("chain is stable", block_hash_chain(ids, BLOCK) == block_hash_chain(list(ids), BLOCK))

    print("state gate")
    record.state = None
    check("record without state does not hit", cache.lookup(ids, BLOCK) is None)
    record.state = object()
    check("state restored hits again", cache.lookup(ids, BLOCK) is record)

    print("page gate")
    broken = dict(record.pages)
    broken.pop(0)
    record.pages = broken
    check("missing logical page does not hit", cache.lookup(ids, BLOCK) is None)

    print("capacity")
    for i in range(4):
        cache.publish(token_ids=[i] * 32, block_size=BLOCK, page_tokens=PAGE,
                      pages=make_pages(32, origin=i), request_id=f"r{i}", state=object(),
                      compacted_in_prefill=False)
    check("lru keeps capacity", len(cache) <= 2, len(cache))

    print("failed" if failed else "all prefix-cache checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
