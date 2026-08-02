import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import LossInputs, compute_loss, opd_loss_fn, setup_loss_fns


def test_topk_opd_unions_teacher_support_with_sample_without_duplicates():
    student_logits = torch.tensor(
        [[1.2, -0.4, 0.8, 0.1, -1.3], [0.2, 1.1, -0.5, 0.4, 0.0]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor([[-0.2, 1.5, 0.3, -0.7, 0.9], [1.4, -0.1, 0.7, -0.8, 0.2]])
    teacher_topk_ids = teacher_logits.topk(2, dim=-1).indices
    sampled_ids = torch.tensor([0, 0])
    assert not (teacher_topk_ids[0] == sampled_ids[0]).any()
    assert (teacher_topk_ids[1] == sampled_ids[1]).any()

    student_logprobs = student_logits.log_softmax(dim=-1)
    teacher_logprobs = teacher_logits.log_softmax(dim=-1)
    student_topk = student_logprobs.gather(-1, teacher_topk_ids)
    teacher_topk = teacher_logprobs.gather(-1, teacher_topk_ids)
    behavior_logprobs = torch.tensor([-0.7, -1.1])

    result = opd_loss_fn(
        LossInputs(
            trainer_logprobs=student_logprobs.gather(-1, sampled_ids[:, None]).squeeze(-1),
            inference_logprobs=behavior_logprobs,
            teacher_logprobs=teacher_logprobs.gather(-1, sampled_ids[:, None]).squeeze(-1),
            advantages=torch.zeros(2),
            loss_mask=torch.ones(2, dtype=torch.bool),
            teacher_topk_logprobs=teacher_topk,
            student_topk_logprobs=student_topk,
            sampled_token_in_teacher_topk=(teacher_topk_ids == sampled_ids[:, None]).any(dim=-1),
        )
    )

    student_p = student_topk.exp()
    expected_topk = (student_p * (student_topk - teacher_topk)).sum(dim=-1)
    student_sampled_logprobs = student_logprobs.gather(-1, sampled_ids[:, None]).squeeze(-1)
    teacher_sampled_logprobs = teacher_logprobs.gather(-1, sampled_ids[:, None]).squeeze(-1)
    expected_sample_term = student_sampled_logprobs.exp() * (student_sampled_logprobs - teacher_sampled_logprobs)
    expected_per_token = expected_topk + torch.tensor([1.0, 0.0]) * expected_sample_term
    expected = expected_per_token.sum()
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(
        result.metrics["topk_student_mass"],
        student_p.sum(dim=-1),
    )
    torch.testing.assert_close(result.metrics["topk_plus_sampled_reverse_kl"], expected_per_token)
    torch.testing.assert_close(result.metrics["sampled_token_in_teacher_topk"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        result.metrics["topk_plus_sampled_student_mass"],
        student_p.sum(dim=-1) + torch.tensor([1.0, 0.0]) * student_sampled_logprobs.exp(),
    )
    expected_sampled_reverse_kl = behavior_logprobs - teacher_sampled_logprobs
    torch.testing.assert_close(result.metrics["sampled_reverse_kl"], expected_sampled_reverse_kl)
    torch.testing.assert_close(
        result.metrics["teacher_kl"], (teacher_sampled_logprobs - student_sampled_logprobs).mean()
    )

    result.loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_mopd_eq5_uses_only_teacher_topk_and_matches_value_and_gradient():
    student_logits = torch.tensor([[1.2, -0.4, 0.8, 0.1, -1.3]], requires_grad=True)
    teacher_logits = torch.tensor([[-0.2, 1.5, 0.3, -0.7, 0.9]])
    teacher_topk_ids = teacher_logits.topk(2, dim=-1).indices
    sampled_id = torch.tensor([0])
    assert not (teacher_topk_ids == sampled_id[:, None]).any()

    student_logprobs = student_logits.log_softmax(dim=-1)
    teacher_logprobs = teacher_logits.log_softmax(dim=-1)
    student_topk = student_logprobs.gather(-1, teacher_topk_ids)
    teacher_topk = teacher_logprobs.gather(-1, teacher_topk_ids)
    sampled_student = student_logprobs.gather(-1, sampled_id[:, None]).squeeze(-1)
    sampled_teacher = teacher_logprobs.gather(-1, sampled_id[:, None]).squeeze(-1)
    inputs = LossInputs(
        trainer_logprobs=sampled_student,
        inference_logprobs=sampled_student.detach(),
        teacher_logprobs=sampled_teacher,
        advantages=torch.zeros(1),
        loss_mask=torch.ones(1, dtype=torch.bool),
        teacher_topk_logprobs=teacher_topk,
        student_topk_logprobs=student_topk,
        # Eq. 5 intentionally neither requires nor unions the sampled token.
        sampled_token_in_teacher_topk=None,
    )

    result = opd_loss_fn(inputs, topk_objective="mopd_eq5")
    reference = (student_topk.exp() * (student_topk - teacher_topk) - student_topk.exp() + teacher_topk.exp()).sum()
    torch.testing.assert_close(result.loss, reference)
    torch.testing.assert_close(result.metrics["mopd_eq5_topk_divergence"], reference.reshape(1))

    result.loss.backward()
    actual_grad = student_logits.grad.detach().clone()
    student_logits_ref = student_logits.detach().clone().requires_grad_(True)
    student_topk_ref = student_logits_ref.log_softmax(dim=-1).gather(-1, teacher_topk_ids)
    reference_grad_loss = (
        student_topk_ref.exp() * (student_topk_ref - teacher_topk) - student_topk_ref.exp() + teacher_topk.exp()
    ).sum()
    reference_grad_loss.backward()
    torch.testing.assert_close(actual_grad, student_logits_ref.grad)


def test_mopd_eq5_has_zero_value_and_gradient_at_teacher_on_selected_support():
    logits = torch.tensor([[0.5, -0.7, 1.1, 0.2]], requires_grad=True)
    logprobs = logits.log_softmax(dim=-1)
    teacher_topk_ids = logprobs.detach().topk(2, dim=-1).indices
    selected_student = logprobs.gather(-1, teacher_topk_ids)
    selected_teacher = selected_student.detach().clone()
    sampled_id = torch.tensor([0])

    result = opd_loss_fn(
        LossInputs(
            trainer_logprobs=logprobs.gather(-1, sampled_id[:, None]).squeeze(-1),
            inference_logprobs=torch.zeros(1),
            teacher_logprobs=logprobs.detach().gather(-1, sampled_id[:, None]).squeeze(-1),
            advantages=torch.zeros(1),
            loss_mask=torch.ones(1, dtype=torch.bool),
            teacher_topk_logprobs=selected_teacher,
            student_topk_logprobs=selected_student,
        ),
        topk_objective="mopd_eq5",
    )
    torch.testing.assert_close(result.loss, torch.zeros_like(result.loss), atol=1e-7, rtol=0)
    result.loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad), atol=1e-7, rtol=0)


def test_mopd_eq5_full_vocab_matches_dense_reverse_kl():
    student_logits = torch.tensor([[0.3, -0.9, 1.4, 0.2]], requires_grad=True)
    teacher_logits = torch.tensor([[-0.6, 1.2, 0.4, 0.1]])
    student_logprobs = student_logits.log_softmax(dim=-1)
    teacher_logprobs = teacher_logits.log_softmax(dim=-1)
    sampled_id = torch.tensor([2])

    result = opd_loss_fn(
        LossInputs(
            trainer_logprobs=student_logprobs.gather(-1, sampled_id[:, None]).squeeze(-1),
            inference_logprobs=torch.zeros(1),
            teacher_logprobs=teacher_logprobs.gather(-1, sampled_id[:, None]).squeeze(-1),
            advantages=torch.zeros(1),
            loss_mask=torch.ones(1, dtype=torch.bool),
            teacher_topk_logprobs=teacher_logprobs,
            student_topk_logprobs=student_logprobs,
        ),
        topk_objective="mopd_eq5",
    )
    dense_reverse_kl = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum()
    torch.testing.assert_close(result.loss, dense_reverse_kl)

    actual_grad = torch.autograd.grad(result.loss, student_logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(dense_reverse_kl, student_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_mopd_eq5_config_dispatch_and_legacy_default():
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([-0.3]),
        inference_logprobs=torch.tensor([-0.3]),
        teacher_logprobs=torch.tensor([-0.7]),
        advantages=torch.zeros(1),
        loss_mask=torch.ones(1, dtype=torch.bool),
        teacher_topk_logprobs=torch.tensor([[-0.7]]),
        student_topk_logprobs=torch.tensor([[-0.3]]),
        sampled_token_in_teacher_topk=torch.tensor([False]),
    )

    legacy = setup_loss_fns(DefaultLossConfig())["opd"](inputs)
    explicit_legacy = opd_loss_fn(inputs, topk_objective="topk_plus_sampled")
    torch.testing.assert_close(legacy.loss, explicit_legacy.loss)

    configured_eq5 = setup_loss_fns(DefaultLossConfig(opd_top_k_objective="mopd_eq5"))["opd"](inputs)
    explicit_eq5 = opd_loss_fn(inputs, topk_objective="mopd_eq5")
    torch.testing.assert_close(configured_eq5.loss, explicit_eq5.loss)


def test_topk_opd_is_sequence_balanced_across_packed_examples():
    # Give every token the same sparse student/teacher pair. A one-token
    # completion and a three-token completion must therefore have equal total
    # weight in the batch objective, rather than weights 1:3.
    student_logp_short = torch.tensor([[-0.4]], requires_grad=True)
    student_logp_long = torch.tensor([[-0.4], [-0.4], [-0.4]], requires_grad=True)
    teacher_logp_short = torch.full((1, 1), -0.8)
    teacher_logp_long = torch.full((3, 1), -0.8)
    masks = [torch.ones(1, dtype=torch.bool), torch.ones(3, dtype=torch.bool)]

    loss, _ = compute_loss(
        trainer_logprobs=[torch.zeros(1), torch.zeros(3)],
        inference_logprobs=[torch.zeros(1), torch.zeros(3)],
        teacher_logprobs=[torch.zeros(1), torch.zeros(3)],
        advantages=[torch.zeros(1), torch.zeros(3)],
        loss_mask=masks,
        loss_fns=setup_loss_fns(DefaultLossConfig()),
        loss_scale=2,
        training_mode="opd",
        teacher_topk_logprobs=[teacher_logp_short, teacher_logp_long],
        student_topk_logprobs=[student_logp_short, student_logp_long],
        sampled_token_in_teacher_topk=[torch.ones(1, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
        sequence_balance=True,
    )

    per_token = torch.exp(torch.tensor(-0.4)) * 0.4
    torch.testing.assert_close(loss, per_token)

    split_losses = []
    for length, student_logp, teacher_logp in (
        (1, student_logp_short, teacher_logp_short),
        (3, student_logp_long, teacher_logp_long),
    ):
        split_loss, _ = compute_loss(
            trainer_logprobs=[torch.zeros(length)],
            inference_logprobs=[torch.zeros(length)],
            teacher_logprobs=[torch.zeros(length)],
            advantages=[torch.zeros(length)],
            loss_mask=[torch.ones(length, dtype=torch.bool)],
            loss_fns=setup_loss_fns(DefaultLossConfig()),
            loss_scale=2,
            training_mode="opd",
            teacher_topk_logprobs=[teacher_logp],
            student_topk_logprobs=[student_logp],
            sampled_token_in_teacher_topk=[torch.ones(length, dtype=torch.bool)],
            sequence_balance=True,
        )
        split_losses.append(split_loss)
    torch.testing.assert_close(sum(split_losses), loss)

    loss.backward()
    # d[p_s log(p_s/p_t)] / d log(p_s) = p_s * (log(p_s/p_t) + 1).
    # The outer batch mean contributes 1/2, and the long sequence's token mean
    # contributes another 1/3 to each of its tokens.
    expected_short_grad = torch.exp(torch.tensor(-0.4)) * 1.4 / 2
    torch.testing.assert_close(student_logp_short.grad.squeeze(), expected_short_grad)
    torch.testing.assert_close(
        student_logp_long.grad,
        torch.full_like(student_logp_long, expected_short_grad / 3),
    )


def test_mopd_eq5_is_sequence_balanced_across_packed_examples():
    student_logp_short = torch.tensor([[-0.4]], requires_grad=True)
    student_logp_long = torch.tensor([[-0.4], [-0.4], [-0.4]], requires_grad=True)
    teacher_logp_short = torch.full((1, 1), -0.8)
    teacher_logp_long = torch.full((3, 1), -0.8)

    loss, metrics = compute_loss(
        trainer_logprobs=[torch.zeros(1), torch.zeros(3)],
        inference_logprobs=[torch.zeros(1), torch.zeros(3)],
        teacher_logprobs=[torch.zeros(1), torch.zeros(3)],
        advantages=[torch.zeros(1), torch.zeros(3)],
        loss_mask=[torch.ones(1, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
        loss_fns=setup_loss_fns(DefaultLossConfig(opd_top_k_objective="mopd_eq5")),
        loss_scale=2,
        training_mode="opd",
        teacher_topk_logprobs=[teacher_logp_short, teacher_logp_long],
        student_topk_logprobs=[student_logp_short, student_logp_long],
        sequence_balance=True,
    )

    student_p = torch.exp(torch.tensor(-0.4))
    teacher_p = torch.exp(torch.tensor(-0.8))
    per_token = student_p * 0.4 - student_p + teacher_p
    torch.testing.assert_close(loss, per_token)
    torch.testing.assert_close(metrics["mopd_eq5_sequence_mean"], torch.full((2,), per_token))

    loss.backward()
    expected_short_grad = student_p * 0.4 / 2
    torch.testing.assert_close(student_logp_short.grad.squeeze(), expected_short_grad)
    torch.testing.assert_close(
        student_logp_long.grad,
        torch.full_like(student_logp_long, expected_short_grad / 3),
    )


def test_sequence_balancing_ignores_padding_only_sequence():
    student_logp = torch.tensor([[-0.4]], requires_grad=True)
    loss, _ = compute_loss(
        trainer_logprobs=[torch.zeros(1), torch.zeros(2)],
        inference_logprobs=[torch.zeros(1), torch.zeros(2)],
        teacher_logprobs=[torch.zeros(1), torch.zeros(2)],
        advantages=[torch.zeros(1), torch.zeros(2)],
        loss_mask=[torch.ones(1, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)],
        loss_fns=setup_loss_fns(DefaultLossConfig()),
        loss_scale=1,
        training_mode="opd",
        teacher_topk_logprobs=[torch.full((1, 1), -0.8), torch.full((2, 1), -1e9)],
        student_topk_logprobs=[student_logp, torch.full((2, 1), -1e9)],
        sampled_token_in_teacher_topk=[torch.ones(1, dtype=torch.bool), torch.ones(2, dtype=torch.bool)],
        sequence_balance=True,
    )

    expected = torch.exp(torch.tensor(-0.4)) * 0.4
    torch.testing.assert_close(loss, expected)
