"""Versioned, invocation-frozen role instructions; evidence cannot replace them."""

from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from symplex.core.contracts import digest

ROOT = Path(__file__).parent
_ACTIVE = ContextVar('symplex_prompt_snapshot', default=None)


def _files():
    return {p.stem: p.read_text() for p in sorted(ROOT.glob('*.md'))}


def snapshot():
    """Copy the active invocation's instructions without changing installed files."""
    return dict(_ACTIVE.get() if _ACTIVE.get() is not None else _files())


@contextmanager
def prompt_overlay(overrides, *, base=None):
    """Evaluate a host-selected frozen variant in this context only; always restore."""
    from symplex.core.contracts import Invalid
    frozen = dict(snapshot() if base is None else base)
    if not isinstance(overrides, dict) or not set(overrides) <= set(frozen):
        raise Invalid('Prompt overlay references an unknown role')
    for name, value in {**frozen, **overrides}.items():
        if (not isinstance(name, str) or not name.replace('_', '').isalnum()
                or not isinstance(value, str) or not value.strip() or len(value) > 16000):
            raise Invalid('Prompt overlays require bounded named instructions')
    frozen.update(overrides)
    token = _ACTIVE.set(frozen)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def prompt(name):
    if not name.replace('_', '').isalnum():
        raise ValueError('Invalid prompt name')
    frozen = _ACTIVE.get()
    return (frozen[name] if frozen is not None else (ROOT / (name + '.md')).read_text()).strip()


def manifest():
    return [
        {'name': name, 'digest': digest(text), 'path': 'agents/prompts/' + name + '.md'}
        for name, text in (_ACTIVE.get() or _files()).items()
    ]


def frozen_prompts(function):
    """Nested tools inherit the same snapshot; concurrent runs are isolated."""
    @wraps(function)
    def invoke(*args, **kwargs):
        if _ACTIVE.get() is not None:
            return function(*args, **kwargs)
        token = _ACTIVE.set(_files())
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return invoke
