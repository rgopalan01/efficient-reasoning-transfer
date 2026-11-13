from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scoring import build_prompt


@dataclass(frozen=True)
class GenerationResult:
    completion: str
    completion_ids: list[int]
    completion_tokens: int
    truncated: bool
    finish_reason: str
    latency_seconds: float


def _training_imports() -> tuple[Any, ...]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are not installed. Install the project with '.[train]'."
        ) from error
    return torch, LoraConfig, PeftModel, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_config: dict[str, Any]) -> Any:
    *_, auto_tokenizer = _training_imports()
    tokenizer = auto_tokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"], use_fast=True
    )
    if tokenizer.chat_template is None:
        raise ValueError("The pinned tokenizer does not provide a native chat template")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy(
    model_config: dict[str, Any], *, adapter_path: str | Path | None = None
) -> tuple[Any, Any]:
    (
        torch,
        _lora_config,
        peft_model,
        _task_type,
        _get_peft_model,
        auto_model,
        auto_tokenizer,
    ) = _training_imports()
    if model_config["quantization"] != "none":
        raise ValueError("Only unquantized loading is supported in version 1")
    tokenizer = auto_tokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"], use_fast=True
    )
    if tokenizer.chat_template is None:
        raise ValueError("The pinned tokenizer does not provide a native chat template")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {
        "revision": model_config["revision"],
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    attention = model_config.get("attention_implementation")
    if attention:
        load_kwargs["attn_implementation"] = attention
    model = auto_model.from_pretrained(model_config["id"], **load_kwargs)
    if adapter_path is not None:
        model = peft_model.from_pretrained(model, str(adapter_path), is_trainable=False)
    return model, tokenizer


def attach_lora(model: Any, adapter_config: dict[str, Any]) -> Any:
    _, lora_config, _, task_type, get_peft_model, *_ = _training_imports()
    targets = list(adapter_config["target_modules"])
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(targets) - available)
    if missing:
        raise ValueError(f"LoRA target modules are missing from the model: {missing}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    config = lora_config(
        task_type=task_type.CAUSAL_LM,
        inference_mode=False,
        r=int(adapter_config["rank"]),
        lora_alpha=int(adapter_config["alpha"]),
        lora_dropout=float(adapter_config["dropout"]),
        target_modules=targets,
        bias="none",
    )
    wrapped = get_peft_model(model, config)
    trainable = [
        (name, parameter.numel())
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Unexpected trainable parameters after attaching LoRA")
    return wrapped


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    total = sum(parameter.numel() for parameter in model.parameters())
    count = sum(trainable.values())
    return {
        "trainable_names": sorted(trainable),
        "trainable_parameters": count,
        "total_parameters": total,
        "trainable_fraction": count / total if total else 0.0,
    }


def generate_completion(
    model: Any,
    tokenizer: Any,
    question: str,
    generation_config: dict[str, Any],
    *,
    prompt_variant: str,
    seed: int,
) -> GenerationResult:
    torch, *_ = _training_imports()
    prompt = build_prompt(question, variant=prompt_variant)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    maximum = int(generation_config["max_new_tokens"])
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=bool(generation_config["do_sample"]),
            temperature=float(generation_config["temperature"]),
            top_p=float(generation_config["top_p"]),
            top_k=int(generation_config["top_k"]),
            max_new_tokens=maximum,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - started
    prompt_length = int(inputs["input_ids"].shape[-1])
    completion_ids = outputs[0, prompt_length:].tolist()
    eos_id = tokenizer.eos_token_id
    ended_with_eos = bool(completion_ids and eos_id is not None and completion_ids[-1] == eos_id)
    truncated = len(completion_ids) >= maximum and not ended_with_eos
    finish_reason = "length" if truncated else "eos" if ended_with_eos else "stop"
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return GenerationResult(
        completion=completion,
        completion_ids=completion_ids,
        completion_tokens=len(completion_ids),
        truncated=truncated,
        finish_reason=finish_reason,
        latency_seconds=latency,
    )


def gpu_hardware() -> dict[str, Any]:
    torch, *_ = _training_imports()
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device": properties.name,
        "vram_bytes": properties.total_memory,
        "cuda_version": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
