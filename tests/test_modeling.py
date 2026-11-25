from __future__ import annotations

from types import SimpleNamespace

import pytest

from effreason.modeling import _resolve_torch_dtype


def test_model_dtype_is_resolved_without_importing_torch():
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )
    assert _resolve_torch_dtype(fake_torch, "float16") is fake_torch.float16
    assert _resolve_torch_dtype(fake_torch, "bfloat16") is fake_torch.bfloat16


def test_model_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported model dtype"):
        _resolve_torch_dtype(SimpleNamespace(), "int8")
