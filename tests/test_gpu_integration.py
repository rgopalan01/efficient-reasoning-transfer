from __future__ import annotations

import os

import pytest


@pytest.mark.gpu
def test_tiny_lora_updates_save_reload_and_resume(tmp_path):
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 on a CUDA host")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from effreason.modeling import attach_lora

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = transformers.Qwen3ForCausalLM(config).cuda()
    base_snapshot = {name: value.detach().clone() for name, value in model.named_parameters()}
    model = attach_lora(
        model,
        {
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    )
    adapter_snapshot = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    inputs = torch.randint(0, 64, (2, 8), device="cuda")
    loss = model(input_ids=inputs, labels=inputs).loss
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(value.detach(), adapter_snapshot[name])
        for name, value in model.named_parameters()
        if name in adapter_snapshot
    )
    checked_base = 0
    for name, parameter in model.base_model.model.named_parameters():
        if name in base_snapshot:
            assert torch.equal(parameter.detach(), base_snapshot[name])
            checked_base += 1
    assert checked_base > 0
    checkpoint_path = tmp_path / "training-state.pt"
    torch.save({"optimizer": optimizer.state_dict()}, checkpoint_path)
    adapter_path = tmp_path / "adapter"
    model.save_pretrained(adapter_path)
    resumed_base = transformers.Qwen3ForCausalLM(config).cuda()
    resumed_base.load_state_dict(base_snapshot)
    reloaded = peft.PeftModel.from_pretrained(
        resumed_base, adapter_path, is_trainable=True
    ).cuda()
    assert any("lora_" in name for name, _ in reloaded.named_parameters())
    resumed_optimizer = torch.optim.AdamW(
        (parameter for parameter in reloaded.parameters() if parameter.requires_grad), lr=1e-3
    )
    state = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_loss = reloaded(input_ids=inputs, labels=inputs).loss
    resumed_loss.backward()
    resumed_optimizer.step()
