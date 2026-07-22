"""Baseline and Thompson Sampling helpers for the Datathon bandit simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTEXT_NUM = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "never_contacted",
    "previous_success",
]

CONTEXT_CAT = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "month",
    "day_of_week",
    "poutcome",
    "campaign_bucket",
]

DEFAULT_ARMS = ("cellular", "telephone")


def build_reward_model(train: pd.DataFrame) -> Pipeline:
    """Train P(y | context, contact) used as counterfactual reward simulator."""
    features = CONTEXT_NUM + CONTEXT_CAT + ["contact"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTEXT_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CONTEXT_CAT + ["contact"]),
        ]
    )
    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(train[features], train["y"])
    return model


def predict_arm_probabilities(
    model: Pipeline,
    contexts: pd.DataFrame,
    arms: Sequence[str] = DEFAULT_ARMS,
) -> Dict[str, np.ndarray]:
    features = CONTEXT_NUM + CONTEXT_CAT + ["contact"]
    probs: Dict[str, np.ndarray] = {}
    for arm in arms:
        tmp = contexts.copy()
        tmp["contact"] = arm
        probs[arm] = model.predict_proba(tmp[features])[:, 1]
    return probs


@dataclass
class ThompsonSamplingPolicy:
    arms: Sequence[str] = DEFAULT_ARMS
    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for arm in self.arms:
            self.alpha.setdefault(arm, 1.0)
            self.beta.setdefault(arm, 1.0)

    def select_arm(self, rng: np.random.Generator) -> str:
        samples = {arm: rng.beta(self.alpha[arm], self.beta[arm]) for arm in self.arms}
        return max(samples, key=samples.get)

    def select_greedy(self) -> str:
        means = self.mean_estimates()
        return max(means, key=means.get)

    def update(self, arm: str, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def mean_estimates(self) -> Dict[str, float]:
        return {
            arm: self.alpha[arm] / (self.alpha[arm] + self.beta[arm]) for arm in self.arms
        }

    def to_dict(self) -> dict:
        return {
            "arms": list(self.arms),
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "mean_estimates": self.mean_estimates(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ThompsonSamplingPolicy":
        return cls(
            arms=payload.get("arms", list(DEFAULT_ARMS)),
            alpha={k: float(v) for k, v in payload.get("alpha", {}).items()},
            beta={k: float(v) for k, v in payload.get("beta", {}).items()},
        )


@dataclass
class SimulationResult:
    policy_name: str
    rewards: List[float]
    choices: List[str]
    exploration_flags: List[int]

    @property
    def conversion(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def exploration_rate(self) -> float:
        return float(np.mean(self.exploration_flags)) if self.exploration_flags else 0.0

    @property
    def cumulative_conversion(self) -> np.ndarray:
        rewards = np.asarray(self.rewards, dtype=float)
        return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)

    def arm_share(self) -> Dict[str, float]:
        if not self.choices:
            return {}
        return pd.Series(self.choices).value_counts(normalize=True).to_dict()


def run_fixed_policy(
    policy_name: str,
    arm: str,
    arm_probs: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n_rounds: Optional[int] = None,
) -> SimulationResult:
    n = n_rounds or len(next(iter(arm_probs.values())))
    rewards: List[float] = []
    choices: List[str] = []
    for i in range(n):
        prob = arm_probs[arm][i % len(arm_probs[arm])]
        reward = float(rng.random() < prob)
        rewards.append(reward)
        choices.append(arm)
    return SimulationResult(policy_name, rewards, choices, [0] * n)


def run_random_policy(
    arms: Sequence[str],
    arm_probs: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n_rounds: Optional[int] = None,
) -> SimulationResult:
    n = n_rounds or len(next(iter(arm_probs.values())))
    rewards: List[float] = []
    choices: List[str] = []
    for i in range(n):
        arm = arms[int(rng.integers(0, len(arms)))]
        prob = arm_probs[arm][i % len(arm_probs[arm])]
        rewards.append(float(rng.random() < prob))
        choices.append(arm)
    return SimulationResult("random", rewards, choices, [0] * n)


def run_thompson_sampling(
    arm_probs: Dict[str, np.ndarray],
    rng: np.random.Generator,
    arms: Sequence[str] = DEFAULT_ARMS,
    n_rounds: Optional[int] = None,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[SimulationResult, ThompsonSamplingPolicy]:
    n = n_rounds or len(next(iter(arm_probs.values())))
    policy = ThompsonSamplingPolicy(
        arms=arms,
        alpha={arm: alpha0 for arm in arms},
        beta={arm: beta0 for arm in arms},
    )
    rewards: List[float] = []
    choices: List[str] = []
    exploration_flags: List[int] = []

    for i in range(n):
        means = policy.mean_estimates()
        greedy_arm = max(means, key=means.get)
        arm = policy.select_arm(rng)
        exploration_flags.append(int(arm != greedy_arm))
        prob = arm_probs[arm][i % len(arm_probs[arm])]
        reward = float(rng.random() < prob)
        policy.update(arm, reward)
        rewards.append(reward)
        choices.append(arm)

    result = SimulationResult(
        "thompson_sampling", rewards, choices, exploration_flags
    )
    return result, policy
