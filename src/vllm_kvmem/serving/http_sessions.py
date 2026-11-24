"""Reject unsupported namespace sizes before they reach the shared scheduler.

Opted in through vLLM's public --middleware argument. Request-body replay and
streaming/disconnect handling belong to Starlette's native HTTP middleware.
"""
import json
import os

from .session_control import session_namespace, session_release_intent
from ..core import settings as _settings


def validate_session_payload(payload, header):
    if not isinstance(payload, dict):
        return  # Native request-schema validation owns malformed payloads.
    # Match GenerateBaseServing._get_session_id in the pinned vLLM version:
    # body, then header, then a nonempty string in vllm_xargs.
    session = payload.get('session_id') or header
    if not session:
        extras = payload.get('vllm_xargs')
        if isinstance(extras, dict) and isinstance(extras.get('session_id'), str):
            session = extras['session_id']
    session_namespace(session, payload.get('cache_salt'))
    if _settings.get('KVMEM_REQUIRE_OWNED_SESSION') == '1' and not session:
        raise ValueError('Block graph requests require an owned session namespace')
    release = session_release_intent(payload.get('kv_transfer_params'), session,
        payload.get('prompt'), payload.get('max_tokens'))
    if release and payload.get('kvmem_query_span') is not None:
        raise ValueError('Session release cannot execute a Query')
    params = payload.get('kv_transfer_params')
    share = params.get('kvmem_share_prefix') if isinstance(params, dict) else None
    if share is not None and (type(share) is not int or share <= 0 or not session or release):
        # Prefix images (serving/prefix_images.py); the scheduler validates P itself.
        raise ValueError('kvmem_share_prefix requires a session and a positive integer token position')


def auto_session_id(payload, prefix_chars):
    """Derive a STABLE per-conversation namespace from the conversation's first content.

    Why the first message and not a prefix of the whole prompt: KVMem reuses state only
    inside one session namespace (same-namespace requests serialize, and one prompt
    checkpoint is retained per session), so the namespace must stay constant across the
    whole conversation. The conversation's opening (system prompt plus the first user
    message) is exactly that: stable for every turn, different between independent tasks
    even when all of them share the same system prompt and tool definitions.

    Continuations then hit upstream's "same session + exact token prefix" rule for free,
    because turn N+1's prompt starts with turn N's history. `prefix_chars` bounds how much
    of that opening is hashed (default 8192 characters ~ the system prompt and tools).

    Release (`kvmem_release_session`) and prefix sharing (`kvmem_share_prefix`) describe an
    explicit conversation or image, so they never auto-derive.
    """
    params = payload.get('kv_transfer_params')
    if isinstance(params, dict) and (params.get('kvmem_release_session')
                                     or params.get('kvmem_share_prefix') is not None):
        return None
    # 0 (or negative) means "hash the whole opening": truncation would map two tasks that
    # share a long identical head (system prompt + pasted document) to one namespace, and the
    # ledger keeps only one prompt checkpoint per namespace - the second task would evict the
    # first (measured: a second task sharing the first 8 KiB replaced the first session).
    limit = int(prefix_chars or 0)
    if limit <= 0:
        limit = None
    material = None
    messages = payload.get('messages')
    if isinstance(messages, list) and messages:
        opening = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get('content')
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = ''.join(part.get('text') or '' for part in content
                               if isinstance(part, dict))
            else:
                text = ''
            opening.append(text)
            if message.get('role') == 'user' and text:
                break          # the first user message closes the opening
        material = '\x00'.join(opening)
    else:
        prompt = payload.get('prompt')
        if isinstance(prompt, str):
            material = prompt
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
            material = ',' + ','.join(str(t) for t in prompt) + ','
    if not material:
        return None
    import hashlib
    text = material if limit is None else material[:max(64, limit)]
    return 'auto-' + hashlib.sha256(text.encode()).hexdigest()[:24]


_CHAT_TOKENIZER = {}


def _served_tokenizer(app_state):
    """The engine's own tokenizer, so ids match what the chat path renders."""
    if app_state is None:
        return None
    for name in ('openai_serving_chat', 'openai_serving_completion'):
        serving = getattr(app_state, name, None)
        renderer = getattr(serving, 'renderer', None)
        tokenizer = getattr(renderer, 'tokenizer', None)
        if tokenizer is not None:
            return tokenizer
    return None


def chat_prompt_tokens(payload, app_state=None):
    """Token ids of a chat payload, rendered with the served chat template.

    The rolling window plans from the prompt length, so a chat request needs the
    ids a completion request would carry. The served tokenizer renders them.
    """
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        return None
    tokenizer = _served_tokenizer(app_state)
    if tokenizer is None:
        import os as _os
        from transformers import AutoTokenizer
        path = _os.environ.get('KVMEM_TOKENIZER_PATH')
        if not path:
            return None
        if path not in _CHAT_TOKENIZER:
            _CHAT_TOKENIZER[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        tokenizer = _CHAT_TOKENIZER[path]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            **(payload.get('chat_template_kwargs') or {}))
    except (ValueError, TypeError, AttributeError):
        return None
    # A HF tokenizer returns a BatchEncoding for tokenize=True; unwrap it.
    if isinstance(ids, dict) or hasattr(ids, 'input_ids'):
        ids = ids['input_ids'] if not isinstance(ids, dict) else ids.get('input_ids')
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids] if isinstance(ids, list) and ids else None


def normalize_query_plan(payload, session_header=None, app_state=None):
    if not isinstance(payload, dict):
        return False  # Preserve native request-schema errors.
    from .request_plan import FIELDS, RequestPlan
    fields = {k: payload[k] for k in FIELDS if k in payload}
    params = dict(payload.get('kv_transfer_params') or {})
    if fields and 'kvmem_request_plan' in params:
        raise ValueError('Duplicate request plan')
    fields = fields or params.get('kvmem_request_plan')
    config = _settings.get('KVMEM_REQUEST_PLAN_CONFIG')
    if fields is not None and not config:
        raise ValueError('Explicit Query planning is not enabled for this backend')
    if not config:
        return False
    cfg = json.loads(config)
    prompt = payload.get('prompt')
    if not isinstance(prompt, list) or any(type(x) is not int or x < 0 for x in prompt):
        if cfg.get('rolling_active_window', False):
            # A chat payload carries messages, not ids. Render them with the
            # served template so the same rolling-window plan applies instead of
            # sending the request down the unmetered legacy entry.
            prompt = chat_prompt_tokens(payload, app_state)
            if prompt is None:
                raise ValueError('Rolling active window requires a token-ID completion prompt')
        elif fields:
            raise ValueError('Explicit Query coordinates require a token-ID completion prompt')
        else:
            return False  # Native chat remains an independent legacy entry.
    reserve = cfg.get('guided_reserve',0)
    if cfg.get('guided_reselect') == 'prefill':
        reserve = 0  # Private scratch precedes pressure, not the final Query.
    plan = RequestPlan.parse(fields or {}, len(prompt), cfg['block_tokens'], cfg['gen_budget'],reserve)
    if (plan.query is not None and
            plan.replay_end - plan.replay_begin > cfg.get('query_replay_budget', cfg['gen_budget'])):
        raise ValueError('Atomic Query exceeds the configured Query replay budget')
    if plan.query is not None and reserve and plan.replay_end + reserve > cfg['max_model_len']:
        raise ValueError('Guided Query scratch exceeds the model position limit')
    max_tokens = payload.get('max_tokens')
    if max_tokens is None:
        max_tokens = payload.get('max_completion_tokens')
    if max_tokens is None:
        max_tokens = 16
        payload['max_tokens'] = max_tokens
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    # G is physical decode headroom in rolling mode, not a response limit.
    # Keep one native request (and its sampler/tool frame) across refreshes.
    # Native serving still enforces prompt + output against max_model_len.
    if max_tokens > cfg['gen_budget'] and not cfg.get('rolling_active_window', False):
        raise ValueError('Single-turn output must remain within G')
    extras = payload.get('vllm_xargs')
    session = payload.get('session_id') or session_header or (extras.get('session_id') if isinstance(extras,dict) else None)
    if not session or plan.reselect == 'force':
        plan.validate_capacity(len(prompt), cfg['select_budget'], output_tokens=max_tokens,
            sink_blocks=cfg.get('sink_blocks', 0), recent_blocks=cfg.get('recent_blocks', 0),
            rolling=cfg.get('rolling_active_window', False),
            pressure_policy=cfg.get('pressure_policy', 'recency'))
    # A session auto Query can disappear after exact-prefix validation. Its
    # combined capacity check belongs to that owner before any import/allocation.
    output_reserve = 0 if cfg.get('rolling_active_window', False) else max_tokens
    if (plan.prefill_window == 'keep_selected' and not session
            and len(prompt) + output_reserve > cfg['select_budget'] + cfg['gen_budget'] - cfg['block_tokens']):
        raise ValueError('keep_selected requires verified active headroom; this cold prompt exceeds K+G')
    params['kvmem_request_plan'] = fields or {}
    for key in FIELDS:
        payload.pop(key, None)
    payload['kv_transfer_params'] = params
    return True


async def validate_prompt_session(request, call_next):
    if request.method == 'POST' and request.url.path.rstrip('/') in (
            '/v1/chat/completions', '/v1/completions', '/v1/responses'):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Preserve native malformed-JSON responses.
        else:
            try:
                mutated = False
                if os.environ.get('KVMEM_AUTO_SESSION', '0') == '1':
                    header = request.headers.get('X-Session-ID')
                    extras = payload.get('vllm_xargs')
                    owned = (payload.get('session_id') or header
                             or (extras.get('session_id')
                                 if isinstance(extras, dict) else None))
                    if not owned:
                        derived = auto_session_id(
                            payload, os.environ.get('KVMEM_AUTO_SESSION_PREFIX_CHARS', '8192'))
                        if derived:
                            payload['session_id'] = derived
                            mutated = True
                validate_session_payload(payload, request.headers.get('X-Session-ID'))
                if normalize_query_plan(payload, request.headers.get('X-Session-ID'),
                                        getattr(request.app, 'state', None)):
                    mutated = True
                if mutated:
                    request._body = json.dumps(payload).encode()
                    request._json = payload
            except ValueError as error:
                from starlette.responses import JSONResponse
                return JSONResponse({'error': {'message': str(error),
                    'type': 'BadRequestError', 'param': 'session_id/cache_salt',
                    'code': 400}}, status_code=400)
    return await call_next(request)
