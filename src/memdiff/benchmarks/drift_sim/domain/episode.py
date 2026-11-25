"""Deterministic procedural state-machine simulator for controlled belief-revision stress tests.

Code owns 100% of ground-truth state transitions, superseded values, and durable rules.
This enables exact computation of:
  - Stale Belief Rate: % of errors matching a superseded historical value
  - Durable Instruction Preservation: retention of long-lived rules across sessions
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field


class DriftSimConfig(BaseModel):
    seed: int = 42
    num_sessions: int = 24
    num_entities: int = 8
    contradiction_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    noise_events_per_session: int = 2
    num_durable_rules: int = 4


class DriftSession(BaseModel):
    session_id: str
    step: int
    events: list[str]
    mutated_keys: list[str] = Field(default_factory=list)
    introduced_rules: list[str] = Field(default_factory=list)

    def to_trajectory_dict(self) -> dict[str, object]:
        """Adapt a DriftSim session to the same trajectory shape as LongMemEval-V2."""
        return {
            "id": self.session_id,
            "domain": "drift_sim",
            "environment": "ops_state_machine",
            "goal": f"Process operational updates for {self.session_id}",
            "outcome": "success",
            "start_url": f"drift://session/{self.step}",
            "states": [
                {
                    "state_index": idx,
                    "step": idx,
                    "url": f"drift://session/{self.step}/event/{idx}",
                    "action": None,
                    "thought": None,
                    "accessibility_tree": text,
                }
                for idx, text in enumerate(self.events)
            ],
        }


class DriftQuery(BaseModel):
    id: str
    query_type: Literal["dynamic_state", "durable_rule", "premise_absence"]
    entity: str
    question: str
    gold_answer: str
    stale_answers: list[str] = Field(
        default_factory=list,
        description="Historically valid values that were later superseded; retrieving these counts as a stale belief error.",
    )


class DriftEpisode(BaseModel):
    config: DriftSimConfig
    sessions: list[DriftSession]
    queries: list[DriftQuery]


_SERVICES = [
    ("billing-api", "deployment_region", ["us-east-1", "us-west-2", "eu-central-1", "ap-northeast-1"]),
    ("auth-gateway", "token_ttl_minutes", ["15", "30", "45", "60"]),
    ("checkout-svc", "oncall_owner", ["Alice Kim", "Brian Patel", "Clara Chen", "David Okafor"]),
    ("search-index", "replica_count", ["2", "3", "5", "8"]),
    ("payments-worker", "retry_policy", ["exponential-3x", "linear-5x", "jitter-4x", "circuit-breaker"]),
    ("analytics-etl", "warehouse_target", ["snowflake-prod", "bigquery-v2", "duckdb-lake", "clickhouse-eu"]),
    ("notification-hub", "rate_limit_rpm", ["120", "300", "600", "1200"]),
    ("feature-flags", "primary_store", ["redis-cluster-a", "redis-cluster-b", "etcd-v3", "postgres-ro"]),
    ("ml-router", "fallback_model", ["gpt-4o-mini", "gpt-5-mini", "llama-3.3-70b", "qwen-2.5-72b"]),
    ("audit-ledger", "retention_days", ["30", "90", "180", "365"]),
]

_DURABLE_RULES = [
    (
        "change_freeze",
        "Never run schema migrations on Fridays after 14:00 UTC without VP approval.",
        "Fridays after 14:00 UTC",
    ),
    (
        "pii_logging",
        "All customer email addresses must be SHA-256 hashed before writing to application logs.",
        "SHA-256 hashed",
    ),
    (
        "canary_gate",
        "Production deploys to checkout-svc require 15 minutes at 5% canary traffic before full rollout.",
        "5% canary traffic for 15 minutes",
    ),
    (
        "rollback_sla",
        "If p99 latency exceeds 800ms for 3 consecutive minutes, trigger automated rollback immediately.",
        "800ms for 3 consecutive minutes",
    ),
    (
        "secret_rotation",
        "Database credentials must only be provisioned via Vault dynamic leases, never static env vars.",
        "Vault dynamic leases",
    ),
]

_NOISE_TEMPLATES = [
    "Temporary health-check spike on pod-{pod} (CPU {cpu}%) resolved automatically after 12 seconds.",
    "One-off debug request req-{req} from test harness returned HTTP 200 in {ms}ms.",
    "Transient staging build #{build} queued behind nightly lint check.",
    "Ephemeral preview branch pr-{pr} cleaned up after merge.",
]


def generate_episode(cfg: DriftSimConfig) -> DriftEpisode:
    rng = random.Random(cfg.seed)

    selected_entities = _SERVICES[: min(cfg.num_entities, len(_SERVICES))]
    selected_rules = _DURABLE_RULES[: min(cfg.num_durable_rules, len(_DURABLE_RULES))]

    current_state: dict[str, str] = {}
    history: dict[str, list[str]] = {}
    sessions: list[DriftSession] = []

    for step in range(1, cfg.num_sessions + 1):
        events: list[str] = []
        mutated: list[str] = []
        introduced_rules: list[str] = []

        # Introduce durable rules early so we can test long-horizon retention
        if step <= len(selected_rules):
            rule_key, rule_text, _ = selected_rules[step - 1]
            events.append(f"[DURABLE POLICY:{rule_key}] {rule_text}")
            introduced_rules.append(rule_key)

        # Initialize unseen entities first, then apply controlled contradictions
        if step == 1:
            for svc, attr, domain_vals in selected_entities:
                val = rng.choice(domain_vals)
                key = f"{svc}.{attr}"
                current_state[key] = val
                history[key] = [val]
                events.append(f"[STATE INIT] {svc} {attr} is set to {val}.")
                mutated.append(key)
        else:
            for svc, attr, domain_vals in selected_entities:
                key = f"{svc}.{attr}"
                if rng.random() < cfg.contradiction_rate:
                    old_val = current_state[key]
                    candidates = [v for v in domain_vals if v != old_val]
                    new_val = rng.choice(candidates)
                    current_state[key] = new_val
                    history[key].append(new_val)
                    events.append(
                        f"[STATE UPDATE] {svc} {attr} changed from {old_val} to {new_val} (replaces previous setting)."
                    )
                    mutated.append(key)

        # Add temporary noise events that should be ignored/forgotten
        for _ in range(cfg.noise_events_per_session):
            tmpl = rng.choice(_NOISE_TEMPLATES)
            events.append(
                "[TEMP LOG] "
                + tmpl.format(
                    pod=rng.randint(100, 999),
                    cpu=rng.randint(45, 92),
                    req=rng.randint(10000, 99999),
                    ms=rng.randint(15, 420),
                    build=rng.randint(1000, 9999),
                    pr=rng.randint(10, 499),
                )
            )

        sessions.append(
            DriftSession(
                session_id=f"drift_s{step:03d}",
                step=step,
                events=events,
                mutated_keys=mutated,
                introduced_rules=introduced_rules,
            )
        )

    queries: list[DriftQuery] = []
    for idx, (svc, attr, _) in enumerate(selected_entities, start=1):
        key = f"{svc}.{attr}"
        latest = current_state[key]
        stale = sorted({v for v in history[key][:-1] if v != latest})
        queries.append(
            DriftQuery(
                id=f"dq_state_{idx:02d}",
                query_type="dynamic_state",
                entity=key,
                question=f"What is the current `{attr}` setting for `{svc}`?",
                gold_answer=latest,
                stale_answers=stale,
            )
        )

    for idx, (rule_key, _rule_text, expected_phrase) in enumerate(selected_rules, start=1):
        queries.append(
            DriftQuery(
                id=f"dq_rule_{idx:02d}",
                query_type="durable_rule",
                entity=rule_key,
                question=f"According to durable operational policy `{rule_key}`, what is the required rule or threshold?",
                gold_answer=expected_phrase,
                stale_answers=[],
            )
        )

    return DriftEpisode(config=cfg, sessions=sessions, queries=queries)
