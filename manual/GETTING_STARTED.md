# 실행 가이드: 벤치마크를 돌리기 전에 직접 써 보기

아래 단계는 쉬운 것부터 어려운 것 순서입니다. 0~2단계는 GPU나 LLM 없이 몇 초 만에 끝납니다.

| 단계 | 필요한 것 | 걸리는 시간 |
|---|---|---|
| 0. 설치 | `uv`, (뷰어용) `pnpm` | 1분 |
| 1. 오프라인 데모 | 없음 | 수 초 |
| 2. 결과 들여다보기 (sqlite3, 웹 뷰어) | 없음 | – |
| 3. 실제 LLM으로 데모 | Ollama + 모델 2개 | 수 분 |
| 4. 내 데이터를 파이썬에서 직접 넣기 | Ollama | – |
| 5. 작은 벤치마크 스모크 런 | Ollama (+ LME-V2 데이터) | 수 분~수십 분 |

## 0. 설치

```bash
uv sync
cp .env.example .env      # 3단계부터 필요
uv run pytest -q          # 전부 오프라인 테스트라 바로 통과해야 정상
```

## 1. 오프라인 데모: 메모리 엔진이 하는 일을 눈으로 확인

```bash
uv run python scripts/demo_memory.py
```

LLM 대신 규칙 기반 extractor/reconciler와 해시 임베딩을 씁니다. 하는 일은 다음과 같습니다.

1. DriftSim 세션 5개를 만듭니다. 서비스 설정값 3개가 계속 바뀌고, 규칙 2개와 노이즈 로그가 섞여 있습니다
2. 세션을 하나씩 `BeliefRevisionPolicy.insert()`에 넣고, step마다 ledger에 기록된 연산(`ADD`/`SUPERSEDE`/`IGNORE`)을 출력합니다
3. 최종 활성 belief 목록을 출력합니다. 옛 값은 `superseded`라서 빠져 있습니다
4. 정답이 알려진 질문마다 `query()` top-1 결과와 정답(gold), stale 오답 후보를 같이 보여줍니다
5. `runs/demo/demo.sqlite`(ledger)와 `runs/demo/replay_demo.json`(뷰어용)을 남깁니다

출력 예시(일부):

```text
--- step 5 (drift_s005) input events:
    [STATE UPDATE] billing-api deployment_region changed from ap-northeast-1 to us-west-2 (replaces previous setting).
  ledger ops:
    [SUPERSEDE  ] Superseded b_cd222fc197 -> b_b741cac1f8: newer value for same surface
    [IGNORE     ] Ignored temporary noise on ops.noise: [TEMP LOG] ...

  Q: What is the current `deployment_region` setting for `billing-api`?
  gold='us-west-2' stale=['ap-northeast-1', 'us-east-1'] -> top-1 HIT
```

옵션: `--sessions 10`, `--top-k 5`, `--out-dir runs/demo2`

## 2. 결과 들여다보기

**DB 직접 조회**: 테이블 의미와 쿼리 예시는 [DATABASE.md](DATABASE.md) 5절에 있습니다.

```bash
sqlite3 -header -column runs/demo/demo.sqlite \
  "SELECT step, op, summary FROM timeline_events ORDER BY id"
```

**웹 뷰어 (Memory Time Machine)**:

```bash
pnpm --dir web install
pnpm --dir web dev         # 출력된 http://localhost:5173 열기
```

오른쪽 위 **Load replay JSON**에서 `runs/demo/replay_demo.json`을 고르면, 슬라이더로 step을 움직이며 belief가 생기고 대체되는 과정을 볼 수 있습니다. 벤치마크 실행 결과(`runs/<id>/replay_*.json`)도 같은 방법으로 엽니다.

## 3. 실제 LLM으로 같은 데모 돌리기

Ollama 준비(호스트에서):

```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve      # 12GB VRAM 기준. 기본 4K면 프롬프트가 잘림
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
# 원격 GPU면 Mac에서: ssh -fN -L 11434:127.0.0.1:11434 <gpu-host>
```

연결을 확인한 뒤 실행합니다.

```bash
uv run python scripts/check_ollama.py         # "All configured models are installed and ready!"가 나와야 함
uv run python scripts/demo_memory.py --live --no-think
```

`--live`에서는 extractor와 reconciler를 LLM이 맡습니다(`policies/llm_ops.py`). 오프라인 결과와 비교하면 LLM이 surface 키를 일관되게 잡는지, SUPERSEDE를 제대로 고르는지 바로 보입니다. `--no-think`를 빼면 qwen이 사고 과정을 거쳐서 느리지만 더 정확할 수 있습니다. 응답은 `.cache/llm/llm_cache.db`에 캐시되므로 같은 입력을 다시 돌리면 빠릅니다.

## 4. 내 데이터를 파이썬에서 직접 넣어 보기

trajectory는 그냥 dict입니다. `states[*].accessibility_tree`(또는 `text`)에 관측 텍스트를 넣으면 됩니다.

```python
# uv run python
from memdiff.benchmarks.drift_sim.runner import build_default_policy
from memdiff.core.store import BeliefStore
from memdiff.llm import CachedLLMClient

client = CachedLLMClient()  # .env 설정 사용
with BeliefStore("runs/playground/memory.sqlite") as store:
    policy = build_default_policy("belief_revision", store=store, client=client, think=False)

    policy.insert({
        "id": "note_001", "domain": "notes", "goal": "team standup", "outcome": "success",
        "states": [
            {"state_index": 0, "accessibility_tree": "The staging DB lives in us-east-1."},
            {"state_index": 1, "accessibility_tree": "Rule: never deploy on Fridays."},
        ],
    })
    policy.insert({
        "id": "note_002", "domain": "notes", "goal": "migration update", "outcome": "success",
        "states": [{"state_index": 0, "accessibility_tree": "Staging DB was migrated to eu-west-1."}],
    })

    for item in policy.query("Where is the staging DB?", top_k=3):
        print(item["value"], "\n---")
    for ev in store.list_events():
        print(ev.step, ev.op.value, ev.summary)
    print(policy.stats())
```

- 정책 이름을 `"naive_rag"`, `"recency_rag"`, `"structured_extract"`로 바꾸면 같은 입력으로 베이스라인과 비교할 수 있습니다
- `insert()` 한 번이 step 1입니다. 시간 순서대로 넣으세요
- 특정 시점 상태: `store.get_state_at_step(1)`

## 5. 작은 벤치마크 스모크 런

### Track 2: DriftSim (데이터 다운로드 없음)

```bash
uv run python scripts/run_drift_sim.py \
  --policies naive_rag belief_revision \
  --sessions 6 --entities 3 --durable-rules 2 \
  --no-think --run-id smoke_drift
```

결과는 `runs/smoke_drift/`에 생깁니다.
- `summary.json`: 정책별 `overall_accuracy`, `stale_belief_rate`(낮을수록 좋음), `durable_rule_preservation`(높을수록 좋음), LLM 사용량
- `<policy>_predictions.jsonl`: 질문별 응답, 정답, 검색된 컨텍스트
- `<policy>.sqlite`: 정책별 ledger
- `replay_belief_revision.json`: 뷰어용(naive_rag와의 Q&A 비교 포함)

중간에 끊기면 같은 `--run-id`에 `--resume`을 붙여 다시 실행하세요. 기본 규모는 `--sessions 24 --entities 8`입니다.

### Track 1: LongMemEval-V2 (데이터 약 100 MB 필요)

```bash
uv run python scripts/fetch_lmev2_small.py --domain enterprise --include-question-screenshots
uv run python scripts/run_lmev2.py --policy naive_rag --limit 5 --no-images --no-think --run-id smoke_lme
```

- `--limit N`: 앞에서 N문항만. `--balanced-per-type N`: 유형별로 N문항씩
- 문항이 적어도 haystack trajectory를 통째로 ingest하므로, `belief_revision`/`structured_extract`는 trajectory마다 LLM 추출을 해서 오래 걸립니다. 먼저 `naive_rag`로 파이프라인이 도는지 확인하세요
- 결과: `runs/smoke_lme/<policy>_metrics.json`, `<policy>_predictions.jsonl`, `replay_<policy>.json`

## 자주 막히는 곳

| 증상 | 원인과 해결 |
|---|---|
| `check_ollama.py`가 연결 실패 | `ollama serve` 실행 여부, SSH 터널, `.env`의 `LLM_BASE_URL` 확인 |
| 응답이 빈 문자열이거나 `\boxed{}`가 없음 | 사고 과정이 토큰을 다 써버린 경우입니다. `--no-think`를 쓰거나 `OLLAMA_CONTEXT_LENGTH=32768`을 확인하세요 |
| `belief_revision` 재실행이 느림(캐시 미스) | belief id가 랜덤이라 reconciler 프롬프트가 매번 달라집니다. DriftSim은 `--resume`을 쓰세요 |
| `Missing .../questions.jsonl` | 5단계의 fetch를 먼저 실행하세요 |
