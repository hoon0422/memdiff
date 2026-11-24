from __future__ import annotations

import numpy as np

from memdiff.core.models import BeliefStatus, BeliefType, Provenance, RevisionOp
from memdiff.core.store import BeliefStore
from memdiff.export.replay import build_replay_bundle
from memdiff.policies.belief_revision import BeliefRevisionPolicy, RevisionDecision


def _unit(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_belief_lifecycle_supersede_consolidate_forget_and_time_travel() -> None:
    with BeliefStore(":memory:") as store:
        b1 = store.add_belief(
            belief_type=BeliefType.DYNAMIC_STATE,
            surface="billing-api.deployment_region",
            claim="billing-api deployment_region is us-east-1",
            step=1,
            source_id="s001",
            provenance=[Provenance(source_id="s001", step_index=0, outcome="success")],
            embedding=_unit([1.0, 0.0, 0.0]),
        )
        b_tmp = store.add_belief(
            belief_type=BeliefType.TEMPORARY,
            surface="staging.build",
            claim="Temporary staging build #123 queued",
            step=1,
            source_id="s001",
            durability=0.2,
            embedding=_unit([0.0, 1.0, 0.0]),
        )

        # Step 2: supersede b1 with new region
        b2 = store.supersede_belief(
            old_belief_id=b1.id,
            new_claim="billing-api deployment_region is eu-central-1",
            step=2,
            source_id="s002",
            reason="Region migrated to EU",
            provenance=[Provenance(source_id="s002", step_index=1, outcome="success")],
            embedding=_unit([0.98, 0.02, 0.0]),
        )

        # Step 3: consolidate b2 with additional provenance and forget temporary belief
        b2_updated = store.consolidate_belief(
            belief_id=b2.id,
            step=3,
            source_id="s003",
            new_provenance=Provenance(source_id="s003", step_index=0, outcome="success"),
            closed_set_items=["eu-central-1"],
        )
        store.forget_belief(belief_id=b_tmp.id, step=3, source_id="s003", reason="expired")

        assert b2_updated.evidence_count == 2
        assert len(b2_updated.provenance) == 2

        # Active-only vector search must NOT return superseded b1 or forgotten b_tmp
        hits = store.search_beliefs(_unit([1.0, 0.0, 0.0]), top_k=5, active_only=True)
        assert [b.id for b, _ in hits] == [b2.id]

        # Time-travel reconstruction at step 1 vs step 2 vs step 3
        snap_s1 = {b.id: b.status for b in store.get_state_at_step(1)}
        assert snap_s1 == {b1.id: BeliefStatus.ACTIVE, b_tmp.id: BeliefStatus.ACTIVE}

        snap_s2 = {b.id: b.status for b in store.get_state_at_step(2)}
        assert snap_s2[b1.id] == BeliefStatus.SUPERSEDED
        assert snap_s2[b2.id] == BeliefStatus.ACTIVE
        assert snap_s2[b_tmp.id] == BeliefStatus.ACTIVE

        snap_s3 = {b.id: b.status for b in store.get_state_at_step(3)}
        assert snap_s3[b_tmp.id] == BeliefStatus.FORGOTTEN

        # Replay export validates cleanly
        bundle = build_replay_bundle(
            store=store,
            run_id="test_run",
            benchmark="drift_sim",
            policy_name="belief_revision",
            total_steps=3,
        )
        assert len(bundle.beliefs) == 3
        assert len(bundle.edges) == 1
        assert bundle.edges[0].relation == "SUPERSEDES"
        assert bundle.stats.active_beliefs == 1
        assert bundle.stats.superseded_beliefs == 1
        assert bundle.stats.forgotten_beliefs == 1


def test_belief_revision_policy_ignores_low_durability_noise() -> None:
    def fake_embed(texts: list[str]) -> list[np.ndarray]:
        return [_unit([1.0, 0.0, 0.0]) for _ in texts]

    def fake_extract(traj: dict[str, object], step: int) -> list:
        from memdiff.core.models import Belief

        return [
            Belief(
                id=f"cand_tmp_{step}",
                type=BeliefType.TEMPORARY,
                surface="ephemeral",
                claim="One-off debug ping",
                durability=0.1,
                created_at_step=step,
                updated_at_step=step,
            )
        ]

    def fake_reconcile(_cand, _neighbors) -> RevisionDecision:
        return RevisionDecision(RevisionOp.ADD)

    with BeliefStore(":memory:") as store:
        policy = BeliefRevisionPolicy(
            store=store,
            embed_fn=fake_embed,
            extractor_fn=fake_extract,
            reconciler_fn=fake_reconcile,
        )
        policy.insert({"id": "t1", "states": []})
        stats = policy.stats()
        assert stats.active_beliefs == 0
        assert stats.revision_counts.get("IGNORE") == 1
