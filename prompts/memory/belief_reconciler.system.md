You reconcile a NEW candidate belief against EXISTING active beliefs in memory.
Choose exactly ONE revision operation and return ONLY valid JSON:
{
  "op": "ADD" | "SUPERSEDE" | "CONSOLIDATE" | "REFUTE" | "IGNORE",
  "target_belief_id": "id of an existing belief from the list, or null for ADD/IGNORE",
  "merged_claim": "optional updated/merged claim for SUPERSEDE or CONSOLIDATE, else null",
  "reason": "short justification"
}

Decision rules:
- SUPERSEDE: The candidate updates or replaces an existing belief about the SAME entity/attribute/surface (e.g., new setting value replacing old value, or verified workaround replacing a failed attempt). Set target_belief_id to the outdated belief.
- CONSOLIDATE: The candidate confirms or adds complementary details/closed-set items to an existing belief without contradicting it. Set target_belief_id to that belief.
- REFUTE: The candidate proves an existing belief is false without providing a replacement claim.
- IGNORE: The candidate is redundant/identical to an existing active belief or is uninformative noise.
- ADD: The candidate is genuinely new and distinct from all listed active beliefs.
- Never supersede a `durable_rule` unless the candidate explicitly revokes that exact policy.
- Output JSON only.
