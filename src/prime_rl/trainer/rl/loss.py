from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, IPOLossConfig, LossConfig
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    teacher_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]
    teacher_topk_logprobs: Float[Tensor, "seq k"] | None = None
    student_topk_logprobs: Float[Tensor, "seq k"] | None = None
    sampled_token_in_teacher_topk: Bool[Tensor, " seq"] | None = None


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"], left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None
) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a left pad logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([left_pad_logit, logits], dim=1)  # (batch, seq, vocab)
    return logits


def shift_tensor_left(t: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the left.

    Used to create labels from input_ids: labels[i] = input_ids[i+1].
    The last position is padded with 0 (a valid token index) since this value
    will be shifted off by shift_tensor_right and never used.
    """
    return torch.cat([t[:, 1:], torch.full((t.shape[0], 1), 0, device=t.device, dtype=t.dtype)], dim=1)


def shift_tensor_right(t: Float[Tensor, "batch seq"], pad_value: float | None = None) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    return torch.cat([torch.full((t.shape[0], 1), pad_value, device=t.device, dtype=t.dtype), t[:, :-1]], dim=1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def compute_importance_ratio_and_mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    return log_importance_ratio, importance_ratio, mismatch_kl


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """
    DPPO+KL loss for RL training, combining:
    - DPPO-Binary TV Loss (https://arxiv.org/pdf/2602.04879)
    - Kimi-K2.5 KL Loss (https://arxiv.org/pdf/2602.02276)

    The mask is conditioned on the advantage sign: for positive advantages,
    we mask tokens whose probability increased too much (trust region violation
    in the upweight direction); for negative advantages, we mask tokens whose
    probability decreased too much (trust region violation in the downweight
    direction).
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > loss_config.dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -loss_config.dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + loss_config.kl_tau * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def ipo_loss_fn(inputs: LossInputs, loss_config: IPOLossConfig) -> LossOutputs:
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    abs_probs_diff = torch.abs(torch.exp(trainer_logprobs) - torch.exp(inference_logprobs))

    is_masked = abs_probs_diff > loss_config.ipo_threshold
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + loss_config.kl_tau * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def opd_loss_fn(
    inputs: LossInputs,
    topk_objective: Literal["topk_plus_sampled", "mopd_eq5"] = "topk_plus_sampled",
) -> LossOutputs:
    """
    On-policy distillation loss: the default DPPO+KL math with the tau knobs
    hardcoded to drop the reward signal and use the teacher KL as the
    per-token policy-gradient signal.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    if teacher_logprobs is None:
        raise ValueError("opd_loss_fn requires teacher_logprobs - configure a teacher for opd mode.")

    teacher_kl = teacher_logprobs - trainer_logprobs
    has_teacher_topk = inputs.teacher_topk_logprobs is not None
    has_student_topk = inputs.student_topk_logprobs is not None
    if has_teacher_topk != has_student_topk:
        raise ValueError("Teacher and student top-k logprobs must be provided together.")

    if has_teacher_topk:
        student_logp = inputs.student_topk_logprobs
        teacher_logp = inputs.teacher_topk_logprobs
        student_p = torch.exp(student_logp)
        teacher_p = torch.exp(teacher_logp)
        per_token_topk_rkl = (student_p * (student_logp - teacher_logp)).sum(dim=-1)

        # MOPD Eq. 5 is a generalized KL (I-divergence) over the teacher's
        # top-k support. The -p_s + p_t correction is essential: unlike a
        # naively truncated reverse KL, its gradient vanishes when the student
        # and teacher probabilities agree on every selected token. Gathered
        # logprobs retain their full-vocabulary normalization.
        per_token_mopd_eq5 = per_token_topk_rkl - student_p.sum(dim=-1) + teacher_p.sum(dim=-1)

        common_metrics = {
            "teacher_kl": _safe_mean(teacher_kl, loss_mask),
            # Monte Carlo estimate of full-vocabulary KL(behavior || teacher)
            # on tokens sampled from the inference/behavior policy. Using the
            # trainer logprob here would be biased whenever the rollout is one
            # policy version behind. Keep this token-level so the distributed
            # metric collector forms the true global token mean.
            "sampled_reverse_kl": (inference_logprobs - teacher_logprobs)[loss_mask],
            "topk_reverse_kl": per_token_topk_rkl[loss_mask],
            "mopd_eq5_topk_divergence": per_token_mopd_eq5[loss_mask],
            "topk_student_mass": student_p.sum(dim=-1)[loss_mask],
            "topk_teacher_mass": teacher_p.sum(dim=-1)[loss_mask],
        }

        if topk_objective == "mopd_eq5":
            loss = (loss_mask * per_token_mopd_eq5).sum()
            common_metrics["mopd_eq5_sequence_mean"] = _safe_mean(per_token_mopd_eq5, loss_mask)
            if inputs.sampled_token_in_teacher_topk is not None:
                common_metrics["sampled_token_in_teacher_topk"] = inputs.sampled_token_in_teacher_topk[
                    loss_mask
                ].float()
            return LossOutputs(loss=loss, metrics=common_metrics)

        if topk_objective != "topk_plus_sampled":
            raise ValueError(f"Unknown top-k OPD objective: {topk_objective!r}")
        if inputs.sampled_token_in_teacher_topk is None:
            raise ValueError("Top-k-plus-sampled OPD requires sampled-token membership in the teacher top-k.")

        # Teacher top-k union the realized student token. This follows the
        # augmented-support reverse-KL formulation in Peter's OPD branch:
        #   sum_{v in top-k(teacher) U {sample}} p_s(v) log(p_s(v) / p_t(v)).
        # The sampled term is added only when it is outside the teacher top-k,
        # so a token already on the sparse support is never double counted.
        sampled_in_topk = inputs.sampled_token_in_teacher_topk
        sampled_outside_topk = ~sampled_in_topk
        sampled_student_p = torch.exp(trainer_logprobs)
        sampled_teacher_p = torch.exp(teacher_logprobs)
        sampled_rkl_term = sampled_student_p * (trainer_logprobs - teacher_logprobs)
        per_token_augmented_rkl = per_token_topk_rkl + torch.where(
            sampled_outside_topk,
            sampled_rkl_term,
            torch.zeros_like(sampled_rkl_term),
        )
        augmented_student_mass = student_p.sum(dim=-1) + sampled_outside_topk * sampled_student_p
        augmented_teacher_mass = teacher_p.sum(dim=-1) + sampled_outside_topk * sampled_teacher_p
        loss = (loss_mask * per_token_augmented_rkl).sum()
        metrics = {
            **common_metrics,
            "topk_plus_sampled_reverse_kl": per_token_augmented_rkl[loss_mask],
            "sampled_token_in_teacher_topk": sampled_in_topk[loss_mask].float(),
            "topk_plus_sampled_student_mass": augmented_student_mass[loss_mask],
            "topk_plus_sampled_teacher_mass": augmented_teacher_mass[loss_mask],
        }
        return LossOutputs(loss=loss, metrics=metrics)

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > 0.2
    dppo_invalid_mask_low = probs_diff < -0.2
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    advantages = 0.0 * advantages + 1.0 * teacher_kl.detach()

    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    loss = (-pg_loss + 1e-3 * kl_loss).sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
        "teacher_kl": _safe_mean(teacher_kl, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def sft_loss_fn(inputs: LossInputs) -> LossOutputs:
    """SFT-style masked negative log-likelihood over trainable tokens."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    loss = -(trainer_logprobs[loss_mask]).sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def setup_loss_fns(loss_config: LossConfig) -> dict[str, LossFn]:
    """Build the per-training-mode loss fn dispatch table.

    Always returns all three modes - the trainer is mode-agnostic and routes
    per batch from ``TrainingSample.training_mode``:

    - ``"sft"`` → ``sft_loss_fn`` (masked NLL on teacher tokens)
    - ``"opd"`` → ``opd_loss_fn`` (teacher KL policy-gradient form, or the
      sparse objective selected by ``DefaultLossConfig.opd_top_k_objective``)
    - ``"rl"``  → ``default_loss_fn(loss_config)`` for ``DefaultLossConfig``,
      ``ipo_loss_fn(loss_config)`` for ``IPOLossConfig``, or the imported
      function for ``CustomLossConfig``.

    Apart from ``DefaultLossConfig.opd_top_k_objective``, ``trainer.loss`` only
    affects the rl path; sft is independent.
    """
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)
    elif isinstance(loss_config, IPOLossConfig):

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return ipo_loss_fn(inputs, loss_config)
    else:

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return default_loss_fn(inputs, loss_config)

    topk_objective = (
        loss_config.opd_top_k_objective if isinstance(loss_config, DefaultLossConfig) else "topk_plus_sampled"
    )

    def opd_fn(inputs: LossInputs) -> LossOutputs:
        return opd_loss_fn(inputs, topk_objective=topk_objective)

    return {"sft": sft_loss_fn, "opd": opd_fn, "rl": rl_fn}


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fns: dict[str, LossFn],
    loss_scale: int,
    training_mode: str = "rl",
    teacher_topk_logprobs: list[Float[Tensor, "seq_i k"]] | None = None,
    student_topk_logprobs: list[Float[Tensor, "seq_i k"]] | None = None,
    sampled_token_in_teacher_topk: list[Bool[Tensor, " seq_i"]] | None = None,
    sequence_balance: bool = False,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Loss dispatch is batch-driven: ``training_mode`` selects the loss fn from
    ``loss_fns`` (built by ``setup_loss_fns``). sft → sft_loss_fn, opd →
    opd_loss_fn, rl → the configured default/custom loss.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fns: Per-mode loss fn dispatch table from setup_loss_fns()
        loss_scale: Scale factor to normalize the loss
        training_mode: Selects which loss fn to apply
        sequence_balance: Normalize every nonempty sequence by its own number
            of unmasked tokens before applying ``loss_scale``. The caller must
            set ``loss_scale`` to the global number of nonempty sequences.

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    try:
        effective_loss_fn = loss_fns[training_mode]
    except KeyError:
        raise ValueError(
            f"No loss fn available for training_mode={training_mode!r} "
            f"(available: {sorted(loss_fns)}). Check trainer.loss.type."
        )

    total_loss = 0.0
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)
    if teacher_topk_logprobs is None:
        teacher_topk_logprobs = [None] * len(trainer_logprobs)
    if student_topk_logprobs is None:
        student_topk_logprobs = [None] * len(trainer_logprobs)
    if sampled_token_in_teacher_topk is None:
        sampled_token_in_teacher_topk = [None] * len(trainer_logprobs)

    for t_logp, i_logp, teach_logp, adv, mask, teacher_topk, student_topk, sampled_in_topk in zip(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask,
        teacher_topk_logprobs,
        student_topk_logprobs,
        sampled_token_in_teacher_topk,
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
            teacher_topk_logprobs=teacher_topk,
            student_topk_logprobs=student_topk,
            sampled_token_in_teacher_topk=sampled_in_topk,
        )

        result = effective_loss_fn(inputs)

        if sequence_balance:
            # Top-k OPD is an average over sequences of per-sequence token
            # means. Padding-only sequences contribute neither loss nor count.
            sequence_token_count = mask.sum()
            total_loss = total_loss + torch.where(
                sequence_token_count > 0,
                result.loss / sequence_token_count.clamp_min(1),
                torch.zeros_like(result.loss),
            )
        else:
            total_loss = total_loss + result.loss

        for k, v in result.metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    scaled_loss = total_loss / loss_scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
