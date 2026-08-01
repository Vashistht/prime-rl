from __future__ import annotations

import types
from typing import TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_final_logit_softcapping

FUSED_CE_IGNORE_INDEX = -100


class PrimeLmOutput(TypedDict, total=False):
    """Output from LM head - a TypedDict so pytree can find tensors for FSDP2 hooks."""

    logits: Tensor | None
    logprobs: Tensor | None
    entropy: Tensor | None
    loss: Tensor | None
    topk_logprobs: Tensor | None


def cast_float_and_contiguous(output: PrimeLmOutput) -> PrimeLmOutput:
    """Convert tensors in PrimeLmOutput to float and make contiguous."""

    def _float_and_contiguous(tensor: Tensor | None) -> Tensor | None:
        return tensor.float().contiguous() if tensor is not None else None

    return PrimeLmOutput(
        logits=_float_and_contiguous(output.get("logits")),
        logprobs=_float_and_contiguous(output.get("logprobs")),
        entropy=_float_and_contiguous(output.get("entropy")),
        loss=output.get("loss"),
        topk_logprobs=_float_and_contiguous(output.get("topk_logprobs")),
    )


class FusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        extra_gather_ids: Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "FusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()  # [N]

        if extra_gather_ids is None:
            logprobs, entropy = _SequenceChunkedLogProbEntropyFn.apply(
                hidden_states, self.weight, labels, inv_t, self.chunk_size
            )
            return PrimeLmOutput(logprobs=logprobs.reshape(b, s), entropy=entropy.reshape(b, s))

        gather_ids = extra_gather_ids.reshape(b * s, -1).long().contiguous()
        combined, entropy = _SequenceChunkedLogProbEntropyTopKFn.apply(
            hidden_states,
            self.weight,
            labels,
            inv_t,
            self.chunk_size,
            gather_ids,
        )
        return PrimeLmOutput(
            logprobs=combined[:, 0].reshape(b, s),
            entropy=entropy.reshape(b, s),
            topk_logprobs=combined[:, 1:].reshape(b, s, -1),
        )


class VanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        extra_gather_ids: Tensor | None = None,
    ) -> PrimeLmOutput:
        # VanillaOutputLinear just returns logits - temperature scaling is done externally in train.py
        return PrimeLmOutput(logits=super().forward(hidden_states))


class FusedCrossEntropyOutputLinear(torch.nn.Linear):
    """Fused lm_head + cross-entropy loss using Liger kernel.

    Avoids materializing the full [N, V] logits tensor by fusing the linear
    projection with the cross-entropy loss computation.
    """

    IGNORE_INDEX = FUSED_CE_IGNORE_INDEX

    def __init__(self, in_features: int, out_features: int, softcap: float | None = None):
        super().__init__(in_features, out_features, bias=False)
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        self.fused_ce = LigerFusedLinearCrossEntropyLoss(
            ignore_index=self.IGNORE_INDEX, reduction="mean", softcap=softcap
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        if labels is None:
            return PrimeLmOutput(logits=super().forward(hidden_states))

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        labels_flat = labels.reshape(b * s).contiguous()
        loss = self.fused_ce(self.weight, hidden_flat, labels_flat)
        return PrimeLmOutput(loss=loss)


class QuackFusedCrossEntropyOutputLinear(torch.nn.Linear):
    """Fused lm_head + cross-entropy loss using quack-kernels.

    Chunks the linear projection and cross-entropy computation to avoid
    materializing the full [N, V] logits tensor, using quack's optimized
    CuTe DSL kernels for CE and GEMM.
    """

    IGNORE_INDEX = FUSED_CE_IGNORE_INDEX

    def __init__(self, in_features: int, out_features: int, chunk_size: int = 4096):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        if labels is None:
            return PrimeLmOutput(logits=super().forward(hidden_states))

        from quack.linear_cross_entropy import chunked_linear_cross_entropy

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        labels_flat = labels.reshape(b * s).contiguous()
        loss = chunked_linear_cross_entropy(
            hidden_flat,
            self.weight,
            labels_flat,
            chunk_size=self.chunk_size,
            ignore_index=self.IGNORE_INDEX,
            reduction="mean",
        )
        return PrimeLmOutput(loss=loss)


def _online_logsumexp_and_weighted_update(
    m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_m = torch.amax(chunk_logits, dim=-1)
    m_new = torch.maximum(m, chunk_m)
    exp_old = torch.exp(m - m_new)

    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    t_new = t * exp_old + (chunk_exp * chunk_logits).sum(dim=-1)
    return m_new, s_new, t_new


class _SequenceChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-token logprobs and entropy by chunking over flattened sequence tokens.
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert inv_temperature.dim() == 1, f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert chunk_size > 0

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            token_count = end - start

            m = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((token_count,), device=device, dtype=torch.float32)
            t = torch.zeros((token_count,), device=device, dtype=torch.float32)
            target_logits = torch.zeros((token_count,), device=device, dtype=torch.float32)

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)

                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    target_logits[mask] = scaled_logits[mask, idx]

            logz_chunk = m + torch.log(s)
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_chunk - (t / s)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz)
        ctx.chunk_size = chunk_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in FusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature, logz = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size

        n, _ = hidden.shape
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    grad_logits[mask, idx] += grad_chunk[mask]
                grad_logits = grad_logits * inv_t_chunk

                grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None


class _SequenceChunkedLogProbEntropyTopKFn(torch.autograd.Function):
    """Chunked full-vocabulary log-softmax with sparse differentiable gathers."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        inv_temperature: torch.Tensor,
        chunk_size: int,
        extra_gather_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden.dim() == 2
        assert weight.dim() == 2
        assert labels.dim() == 1
        assert inv_temperature.dim() == 1
        assert extra_gather_ids.dim() == 2
        assert hidden.shape[0] == labels.shape[0] == inv_temperature.shape[0] == extra_gather_ids.shape[0]
        assert hidden.shape[1] == weight.shape[1]
        assert chunk_size > 0

        device = hidden.device
        num_tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        logprobs = torch.empty(num_tokens, device=device, dtype=torch.float32)
        entropy = torch.empty(num_tokens, device=device, dtype=torch.float32)
        logz = torch.empty(num_tokens, device=device, dtype=torch.float32)
        gathered_logprobs = torch.empty(
            (num_tokens, extra_gather_ids.shape[1]),
            device=device,
            dtype=torch.float32,
        )

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            gather_ids_chunk = extra_gather_ids[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            chunk_tokens = end - start

            running_max = torch.full((chunk_tokens,), float("-inf"), device=device, dtype=torch.float32)
            exp_sum = torch.zeros(chunk_tokens, device=device, dtype=torch.float32)
            weighted_sum = torch.zeros(chunk_tokens, device=device, dtype=torch.float32)
            target_logits = torch.zeros(chunk_tokens, device=device, dtype=torch.float32)
            gather_logits = torch.zeros_like(gather_ids_chunk, dtype=torch.float32)

            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                weight_chunk = weight[vocab_start:vocab_end]
                scaled_logits = (hidden_chunk @ weight_chunk.t()).to(torch.float32) * inv_t_chunk

                running_max, exp_sum, weighted_sum = _online_logsumexp_and_weighted_update(
                    running_max,
                    exp_sum,
                    weighted_sum,
                    scaled_logits,
                )

                label_in_chunk = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(label_in_chunk):
                    local_labels = (labels_chunk[label_in_chunk] - vocab_start).long()
                    target_logits[label_in_chunk] = scaled_logits[label_in_chunk, local_labels]

                gather_in_chunk = (gather_ids_chunk >= vocab_start) & (gather_ids_chunk < vocab_end)
                if torch.any(gather_in_chunk):
                    local_ids = (gather_ids_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1)
                    gather_logits = torch.where(
                        gather_in_chunk,
                        torch.gather(scaled_logits, 1, local_ids),
                        gather_logits,
                    )

            logz_chunk = running_max + torch.log(exp_sum)
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_chunk - weighted_sum / exp_sum
            gathered_logprobs[start:end] = gather_logits - logz_chunk.unsqueeze(-1)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz, extra_gather_ids)
        ctx.chunk_size = chunk_size

        # Keeping two outputs preserves the activation-offloading tracker used
        # by the original fused head. Column 0 is the realized-token logprob.
        combined = torch.cat([logprobs.unsqueeze(-1), gathered_logprobs], dim=-1)
        return combined, entropy

    @staticmethod
    def backward(ctx, grad_combined: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in FusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature, logz, extra_gather_ids = ctx.saved_tensors
        grad_logprobs = grad_combined[:, 0].contiguous()
        grad_topk = grad_combined[:, 1:].contiguous()
        chunk_size: int = ctx.chunk_size

        num_tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            gather_ids_chunk = extra_gather_ids[start:end]
            grad_label_chunk = grad_logprobs[start:end].to(torch.float32)
            grad_topk_chunk = grad_topk[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]

            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                weight_chunk = weight[vocab_start:vocab_end]
                scaled_logits = (hidden_chunk @ weight_chunk.t()).to(torch.float32) * inv_t_chunk
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                total_logprob_grad = grad_label_chunk + grad_topk_chunk.sum(dim=-1)
                grad_logits = -total_logprob_grad.unsqueeze(-1) * probs

                label_in_chunk = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(label_in_chunk):
                    local_labels = (labels_chunk[label_in_chunk] - vocab_start).long()
                    grad_logits[label_in_chunk, local_labels] += grad_label_chunk[label_in_chunk]

                gather_in_chunk = (gather_ids_chunk >= vocab_start) & (gather_ids_chunk < vocab_end)
                if torch.any(gather_in_chunk):
                    local_ids = (gather_ids_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1)
                    grad_logits.scatter_add_(
                        1,
                        local_ids,
                        torch.where(
                            gather_in_chunk,
                            grad_topk_chunk,
                            torch.zeros_like(grad_topk_chunk),
                        ),
                    )

                grad_logits = grad_logits * inv_t_chunk
                grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                grad_weight[vocab_start:vocab_end].add_(grad_logits.to(weight.dtype).t() @ hidden_chunk)

        return grad_hidden, grad_weight, None, None, None, None


def inject_prime_lm_head(
    model: nn.Module,
    chunk_size: int | None = None,
    fused_cross_entropy: bool | str = False,
) -> None:
    """
    Inject a PrimeRL LM head into a model.

    This replaces the model's lm_head and overrides the forward method to use labels
    and temperature for chunked loss computation.

    Args:
        model: The model to wrap.
        chunk_size: When set to an int, uses FusedOutputLinear with sequence-token chunked
            logprob/entropy computation (for RL).
        fused_cross_entropy: Controls fused lm_head + CE loss. Accepts:
            - False: no fusion
            - True or "liger": Liger kernel fusion
            - "quack": quack-kernels fusion (chunked linear + CE with CuTe DSL kernels)
    """
    # Guards so we have nicer error messages when a non-standard model is used
    assert hasattr(model, "model"), f"model doesnt have backbone in model.model:\n{model}"
    assert isinstance(model.model, nn.Module), f"model.model is not a nn.Module: {type(model.model)}\n{model}"
    assert hasattr(model, "lm_head"), f"model doesnt have lm_head in model.lm_head:\n{model}"
    assert isinstance(model.lm_head, nn.Linear), f"model.lm_head is not a nn.Linear: {type(model.lm_head)}\n{model}"
    assert not hasattr(model.lm_head, "bias") or model.lm_head.bias is None, (
        f"model.lm_head.bias is not supported: {model.lm_head}\n{model}"
    )

    logger = get_logger()

    # Check for Gemma-style softcapping - dispatch to specialized implementation.
    final_logit_softcapping = get_final_logit_softcapping(model.config)
    if final_logit_softcapping:
        if fused_cross_entropy == "quack":
            raise ValueError(
                "quack_fused does not support Gemma logit softcapping. "
                "Use loss_impl='liger_fused' or loss_impl='torch' instead."
            )
        if not fused_cross_entropy:
            from prime_rl.trainer.models.layers.lm_head_gemma import inject_gemma_lm_head

            inject_gemma_lm_head(model, chunk_size, final_logit_softcapping)
            return

    # Replace the lm_head with the appropriate wrapper
    old_lm_head = model.lm_head
    if fused_cross_entropy == "quack":
        logger.info("Injecting fused cross-entropy LM head (quack-kernels)")
        model.lm_head = QuackFusedCrossEntropyOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
        )
    elif fused_cross_entropy:
        logger.info("Injecting fused cross-entropy LM head (Liger kernel)")
        model.lm_head = FusedCrossEntropyOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            softcap=final_logit_softcapping,
        )
    elif isinstance(chunk_size, int):
        logger.info(f"Injecting chunked LM head with chunk size {chunk_size}")
        model.lm_head = FusedOutputLinear(
            in_features=old_lm_head.in_features, out_features=old_lm_head.out_features, chunk_size=chunk_size
        )
    else:
        logger.info("Injecting vanilla LM head")
        model.lm_head = VanillaOutputLinear(in_features=old_lm_head.in_features, out_features=old_lm_head.out_features)
    model.lm_head.weight = old_lm_head.weight
    del old_lm_head

    _patch_model_forward(model)


def _patch_model_forward(model: nn.Module) -> None:
    # Patch the forward method to use the new lm_head with labels and temperature
    def new_forward(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        temperature: torch.Tensor | None = None,
        extra_gather_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> PrimeLmOutput:
        # For VLM with images, don't create position_ids - let model compute MRoPE internally
        is_multimodal = kwargs.get("pixel_values") is not None
        if position_ids is None and not is_multimodal:
            reference_tensor = input_ids if input_ids is not None else inputs_embeds
            position_ids = torch.arange(1, reference_tensor.shape[1] + 1, device=reference_tensor.device).unsqueeze(0)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # Slice hidden states for logits_to_keep
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )

        head_kwargs = {}
        if extra_gather_ids is not None:
            head_kwargs["extra_gather_ids"] = extra_gather_ids[:, slice_indices]

        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
            **head_kwargs,
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
