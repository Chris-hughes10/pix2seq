"""Tests for advantage computation and the policy-gradient loss."""

import pytest
import torch

from rl.reinforce import (
    RLTrainingConfig,
    compute_advantages,
    compute_policy_gradient_loss,
)


class TestComputeAdvantages:
    def test_greedy(self):
        rewards = torch.tensor([0.5, 0.7, 0.2, 0.4])  # B=2, K=2
        greedy = torch.tensor([0.6, 0.3])
        adv = compute_advantages(rewards, 2, "greedy", greedy_rewards=greedy)
        expected = torch.tensor([0.5 - 0.6, 0.7 - 0.6, 0.2 - 0.3, 0.4 - 0.3])
        torch.testing.assert_close(adv, expected)

    def test_greedy_requires_greedy_rewards(self):
        with pytest.raises(ValueError, match="greedy_rewards required"):
            compute_advantages(torch.tensor([0.5, 0.7]), 2, "greedy")

    def test_mean(self):
        rewards = torch.tensor([0.2, 0.4, 0.6, 1.0, 0.0, 0.5])  # B=2, K=3
        adv = compute_advantages(rewards, 3, "mean")
        expected = torch.tensor([-0.2, 0.0, 0.2, 0.5, -0.5, 0.0])
        torch.testing.assert_close(adv, expected)

    def test_loo(self):
        rewards = torch.tensor([0.2, 0.4, 0.6, 1.0, 0.0, 0.5])  # B=2, K=3
        adv = compute_advantages(rewards, 3, "loo")
        # loo: r_i - mean(others) == (r_i - mean) * K/(K-1)
        expected = torch.tensor([-0.3, 0.0, 0.3, 0.75, -0.75, 0.0])
        torch.testing.assert_close(adv, expected)

    def test_loo_equals_mean_of_others(self):
        rewards = torch.tensor([1.0, 0.0, 0.5, 0.7])  # B=1, K=4
        adv = compute_advantages(rewards, 4, "loo")
        for i in range(4):
            others = torch.cat([rewards[:i], rewards[i + 1:]])
            torch.testing.assert_close(adv[i], rewards[i] - others.mean())

    def test_grpo(self):
        rewards = torch.tensor([0.2, 0.4, 0.6, 0.8])  # B=1, K=4
        adv = compute_advantages(rewards, 4, "grpo")
        grouped = rewards.reshape(1, 4)
        expected = (grouped - grouped.mean()) / (grouped.std() + 1e-6)
        torch.testing.assert_close(adv, expected.reshape(-1))

    def test_grpo_zero_variance_group_is_finite(self):
        rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
        adv = compute_advantages(rewards, 4, "grpo")
        assert torch.isfinite(adv).all()
        torch.testing.assert_close(adv, torch.zeros(4))

    @pytest.mark.parametrize("advantage_type", ["loo", "mean", "grpo"])
    def test_k1_raises_for_group_baselines(self, advantage_type):
        with pytest.raises(ValueError, match="requires num_samples >= 2"):
            compute_advantages(torch.tensor([0.5]), 1, advantage_type)


class TestRLTrainingConfig:
    def test_top_p_rejected(self):
        with pytest.raises(ValueError, match="top_k/top_p"):
            RLTrainingConfig(top_p=0.4)

    def test_top_k_rejected(self):
        with pytest.raises(ValueError, match="top_k/top_p"):
            RLTrainingConfig(top_k=50)

    def test_group_baseline_requires_k2(self):
        with pytest.raises(ValueError, match="num_samples_per_image >= 2"):
            RLTrainingConfig(advantage_type="loo", num_samples_per_image=1)

    def test_greedy_allows_k1(self):
        config = RLTrainingConfig(advantage_type="greedy", num_samples_per_image=1)
        assert config.num_samples_per_image == 1

    def test_ppo_not_implemented(self):
        with pytest.raises(NotImplementedError):
            RLTrainingConfig(algorithm="ppo")


class TestPolicyGradientLoss:
    def test_hand_computed_value(self):
        log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        advantages = torch.tensor([1.0, -2.0])

        loss, _ = compute_policy_gradient_loss(log_probs, advantages, mask)
        # seq log-probs: [-3.0, -1.5]; loss = -mean([1*-3.0, -2*-1.5]) = 0.0
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_masked_positions_excluded(self):
        log_probs = torch.tensor([[-1.0, -100.0]])
        mask = torch.tensor([[True, False]])
        advantages = torch.tensor([1.0])
        loss, _ = compute_policy_gradient_loss(log_probs, advantages, mask)
        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_no_gradient_through_advantages(self):
        log_probs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        advantages = torch.tensor([2.0], requires_grad=True)
        mask = torch.ones(1, 2, dtype=torch.bool)

        loss, _ = compute_policy_gradient_loss(log_probs, advantages, mask)
        loss.backward()
        assert log_probs.grad is not None
        assert advantages.grad is None

    def test_clip_range_hook_not_implemented(self):
        log_probs = torch.zeros(1, 2)
        with pytest.raises(NotImplementedError):
            compute_policy_gradient_loss(
                log_probs,
                torch.ones(1),
                torch.ones(1, 2, dtype=torch.bool),
                old_per_token_log_probs=log_probs,
                clip_range=0.2,
            )
