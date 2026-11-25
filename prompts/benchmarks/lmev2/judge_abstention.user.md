Evaluate whether the model correctly handled a flawed-premise question.

Question:
$question

Reference answer (the expected flaw-aware conclusion):
$reference

Model full response:
$response

Model extracted final answer:
$final_answer

Scoring rubric:
- Label 1 only if the model clearly identifies the flawed premise and reaches a conclusion equivalent to the reference answer.
- Label 0 if the model follows the flawed premise, gives a generic UNKNOWN without naming the flaw, or contradicts itself.

Output JSON only:
{"label": 0 or 1, "reason": "short rationale"}
