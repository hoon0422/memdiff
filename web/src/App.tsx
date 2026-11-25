import { useMemo, useState } from "react";
import { SAMPLE_REPLAY } from "./data/sampleReplay";
import type { Belief, BeliefStatus, ConsolidatePrevious, ReplayBundle } from "./types/replay";

/** Same rule as reconstruct_state_at_step() in audit/domain/time_travel.py. */
function computeBeliefAtStep(bundle: ReplayBundle, belief: Belief, step: number): Belief | null {
  if (belief.created_at_step > step) return null;
  let status: BeliefStatus = "active";
  let superseded_by: string | null = null;
  let content: Partial<ConsolidatePrevious> | undefined;
  for (const ev of bundle.events) {
    // The first CONSOLIDATE after `step` holds the content as it was at `step`.
    if (ev.op === "CONSOLIDATE" && ev.step > step && ev.belief_id === belief.id && !content) {
      content = ev.details.previous as ConsolidatePrevious | undefined;
    }
    if (ev.step > step || ev.target_belief_id !== belief.id) continue;
    if (ev.op === "SUPERSEDE") {
      status = "superseded";
      superseded_by = ev.belief_id ?? null;
    }
    if (ev.op === "REFUTE") status = "refuted";
    if (ev.op === "FORGET") status = "forgotten";
  }
  return { ...belief, ...content, status, superseded_by };
}

export function App() {
  const [bundle, setBundle] = useState<ReplayBundle>(SAMPLE_REPLAY);
  const [step, setStep] = useState<number>(bundle.total_steps);

  async function loadReplayFile(file: File | undefined) {
    if (!file) return;
    try {
      const loaded = JSON.parse(await file.text()) as ReplayBundle;
      setBundle(loaded);
      setStep(loaded.total_steps);
    } catch (err) {
      alert(`Could not read replay JSON: ${err}`);
    }
  }

  const visibleBeliefs = useMemo(() => {
    return bundle.beliefs
      .map((b) => computeBeliefAtStep(bundle, b, step))
      .filter((b): b is Belief => b !== null);
  }, [bundle, step]);

  const stepEvents = useMemo(
    () => bundle.events.filter((ev) => ev.step === step),
    [bundle, step],
  );

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, sans-serif", maxWidth: 960, margin: "0 auto", padding: 24, color: "#111827" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", borderBottom: "1px solid #e5e7eb", paddingBottom: 12 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24 }}>MemDiff — Memory Time Machine</h1>
          <p style={{ margin: "4px 0 0", color: "#6b7280", fontSize: 14 }}>
            Run: <code>{bundle.run_id}</code> · Benchmark: <code>{bundle.benchmark}</code>
          </p>
        </div>
        <div style={{ fontSize: 14, fontWeight: 600, textAlign: "right" }}>
          Session {step} / {bundle.total_steps}
          <label style={{ display: "block", marginTop: 6, fontSize: 12, fontWeight: 400 }}>
            Load replay JSON:{" "}
            <input type="file" accept=".json,application/json" onChange={(e) => loadReplayFile(e.target.files?.[0])} />
          </label>
        </div>
      </header>

      <section style={{ margin: "20px 0", padding: 16, background: "#f9fafb", borderRadius: 8, border: "1px solid #e5e7eb" }}>
        <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
          Timeline Scrubber (Session 1 → {bundle.total_steps})
        </label>
        <input
          type="range"
          min={1}
          max={bundle.total_steps}
          value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          style={{ width: "100%" }}
        />
        <div style={{ marginTop: 10, fontSize: 13, color: "#374151" }}>
          <strong>Events at Session {step}:</strong>
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {stepEvents.map((ev, i) => (
              <li key={i}>
                <code>[{ev.op}]</code> ({ev.source_id}) — {ev.summary}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, marginBottom: 10 }}>Belief Ledger Snapshot ({visibleBeliefs.length})</h2>
        <div style={{ display: "grid", gap: 10 }}>
          {visibleBeliefs.map((belief) => {
            const statusAtStep = belief.status;
            const isStruck = statusAtStep === "superseded" || statusAtStep === "refuted";
            const isFaded = statusAtStep === "forgotten" || isStruck;
            return (
              <div
                key={belief.id}
                style={{
                  padding: 12,
                  borderRadius: 8,
                  border: "1px solid #e5e7eb",
                  opacity: isFaded ? 0.55 : 1,
                  background: statusAtStep === "active" ? "#ffffff" : "#f3f4f6",
                }}
              >
                <div style={{ display: "flex", gap: 8, fontSize: 12, marginBottom: 4 }}>
                  <code>{belief.id}</code>
                  <span style={{ fontWeight: 600 }}>{belief.type}</span>
                  <span>·</span>
                  <span>{belief.surface}</span>
                  <span style={{ marginLeft: "auto", fontWeight: 700, textTransform: "uppercase" }}>
                    {statusAtStep}
                  </span>
                </div>
                <div style={{ textDecoration: isStruck ? "line-through" : "none", fontSize: 14 }}>
                  {belief.claim}
                </div>
                <div style={{ marginTop: 4, fontSize: 12, color: "#4b5563" }}>
                  Evidence {belief.evidence_count} · confidence {belief.confidence.toFixed(2)} · sources:{" "}
                  {belief.provenance.map((p) => p.source_id).join(", ")}
                </div>
                {belief.closed_set_items.length > 0 && (
                  <div style={{ marginTop: 4, fontSize: 12, color: "#4b5563" }}>
                    Closed-set items: {belief.closed_set_items.join(", ")}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {bundle.comparisons.length > 0 && (
        <section>
          <h2 style={{ fontSize: 18, marginBottom: 10 }}>Side-by-Side Q&amp;A Diff</h2>
          {bundle.comparisons.map((cmp) => (
            <div key={cmp.question_id} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>{cmp.question}</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ padding: 10, borderRadius: 6, background: "#fef2f2", border: "1px solid #fecaca" }}>
                  <div style={{ fontSize: 12, fontWeight: 700 }}>{cmp.baseline_policy} ({cmp.baseline_correct ? "PASS" : "FAIL"})</div>
                  <p style={{ fontSize: 13, margin: "6px 0 0" }}>{cmp.baseline_answer}</p>
                </div>
                <div style={{ padding: 10, borderRadius: 6, background: "#ecfdf5", border: "1px solid #a7f3d0" }}>
                  <div style={{ fontSize: 12, fontWeight: 700 }}>{cmp.proposed_policy} ({cmp.proposed_correct ? "PASS" : "FAIL"})</div>
                  <p style={{ fontSize: 13, margin: "6px 0 0" }}>{cmp.proposed_answer}</p>
                </div>
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
