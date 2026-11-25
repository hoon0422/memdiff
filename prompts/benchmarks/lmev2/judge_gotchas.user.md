Evaluate whether the model answer captures the gotcha insight.

Question:
$question

Reference answer (insight points):
$reference

Model full response:
$response

Model extracted final answer:
$final_answer

Scoring rubric:
- Label 1 if the model includes at least one correct insight point from the reference answer (paraphrase acceptable) without contradicting any reference point.
- Label 0 if direction is wrong, contradictory, or generic without insight.

Output JSON only:
{"label": 0 or 1, "reason": "short rationale"}
