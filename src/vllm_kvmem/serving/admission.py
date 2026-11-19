"""Token admission for aligned semantic prefill and target-only decode refresh."""
from dataclasses import dataclass


def validate_target_batch_budget(block_tokens, gen_budget, batch_tokens, owners, query_tokens,
                                 *, request_planning=False):
    # With explicit request planning, each owner is separately clipped by
    # RequestPlan.admit / QueryPolicy.admit before allocation. The native batch
    # can contain several owners' G windows; it is not one owner's KV extent.
    headroom = gen_budget * (owners if request_planning else 1) + owners * query_tokens
    if not block_tokens <= batch_tokens <= headroom:
        raise ValueError('Bounded target batch budget must fit an aligned block and admitted G headroom')


@dataclass(frozen=True)
class QueryPolicy:
    block_tokens: int
    select_budget: int
    gen_budget: int

    @property
    def capacity(self):
        return self.select_budget + self.gen_budget

    def admit(self, history, desired, prompt_tokens, offset):
        local = history - offset
        if min(history, desired, prompt_tokens, local) < 0 or local > self.capacity:
            raise ValueError("Invalid native history or working-set extent")
        tokens = min(desired, self.capacity - local)
        if history < prompt_tokens:
            tokens = min(tokens, prompt_tokens - history)
            # Install the all-history bootstrap exactly at K before semantic
            # query-conditioned chunks. Non-final chunks preserve checkpoints.
            if history < self.select_budget:
                tokens = min(tokens, self.select_budget - history)
            if history + tokens < prompt_tokens:
                tokens = max(0, (history + tokens) // self.block_tokens * self.block_tokens - history)
        if desired and not tokens and local == self.capacity:
            raise RuntimeError("Working set filled without a preceding query refresh")
        return tokens

    def refresh(self, history, prompt_tokens, offset, lookahead=0):
        if history < self.select_budget or history % self.block_tokens:
            return False
        local = history - offset
        # `lookahead` is the largest step the scheduler can add before the next
        # refresh decision, so triggering at capacity - block - lookahead keeps
        # the native allocation below capacity while rebuilding once per
        # generation budget instead of once per aligned block.
        if history < prompt_tokens:
            return local + max(lookahead, 0) >= self.capacity - self.block_tokens
        # Leave one block for the provisional target query. Speculative batches
        # require their own accepted-boundary checkpoint policy, not this rule.
        return local >= self.capacity - self.block_tokens


def refresh_selection(*, planned, remaining, action, max_batched_tokens):
    """Which picker a refresh uses.

    A plan-less request is ordinary chunked prefill: there is no decode query
    yet, so every refresh before the last chunk keeps the newest window in order
    (`pressure`) instead of running the semantic query-probe selection. The one
    refresh inside the last chunk fills B with the Query selection.
    """
    if planned:
        return ('pressure' if action == 'pressure' else
                'query' if action in ('query', 'guided') else 'chunk')
    # The last chunk's refresh and every decode refresh stay Query-selected.
    return 'chunk' if remaining <= max(1, int(max_batched_tokens)) else 'pressure'
