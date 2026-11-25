from __future__ import annotations

from memdiff.benchmarks.drift_sim.domain.episode import DriftSimConfig, generate_episode


def test_drift_sim_is_deterministic_and_tracks_stale_values() -> None:
    cfg = DriftSimConfig(seed=7, num_sessions=16, num_entities=6, contradiction_rate=0.6)
    ep1 = generate_episode(cfg)
    ep2 = generate_episode(cfg)

    assert ep1.model_dump() == ep2.model_dump()
    assert len(ep1.sessions) == 16

    state_queries = [q for q in ep1.queries if q.query_type == "dynamic_state"]
    rule_queries = [q for q in ep1.queries if q.query_type == "durable_rule"]
    assert len(state_queries) == 6
    assert len(rule_queries) == cfg.num_durable_rules

    # With 16 sessions and 0.6 contradiction rate, dynamic state queries should accumulate stale history
    assert any(len(q.stale_answers) > 0 for q in state_queries)
    for q in state_queries:
        assert q.gold_answer not in q.stale_answers

    # Trajectory adapter produces LongMemEval-V2-compatible shape
    traj = ep1.sessions[0].to_trajectory_dict()
    assert traj["id"] == "drift_s001"
    assert isinstance(traj["states"], list) and len(traj["states"]) > 0
