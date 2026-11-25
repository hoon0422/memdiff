# 데이터베이스 구조: 메모리 ledger와 임베딩

MemDiff는 SQLite 파일 두 종류를 씁니다. 외부 벡터 DB는 쓰지 않습니다.

| DB 파일 | 만드는 곳 | 수명 | 내용 |
|---|---|---|---|
| **메모리 ledger** `runs/<run_id>/<policy>.sqlite` (LME-V2는 `<policy>_group<N>.sqlite`) | `BeliefStore` (`src/memdiff/core/store.py`의 `SCHEMA_SQL`) | 실행 1회(정책 1개)당 1개 | beliefs, belief_edges, timeline_events, chunks |
| **LLM 캐시** `.cache/llm/llm_cache.db` | `CachedLLMClient` (`src/memdiff/llm/openai_client.py`) | 모든 실행이 공유 | chat_cache, embedding_cache |

`BeliefStore(":memory:")`를 쓰면 ledger가 메모리에만 생깁니다(테스트용).

## 1. 메모리 ledger (`BeliefStore`)

```text
             belief_edges                        timeline_events
     (source_id ──relation──► target_id)    (step마다 무슨 연산이 있었나)
                 │      │                         │           │
                 ▼      ▼                         ▼           ▼
 ┌──────────────────────────────────────┐   belief_id   target_belief_id
 │ beliefs                              │◄──────┴───────────┘
 │  id, type, surface, claim, status,   │
 │  superseded_by ──► beliefs.id        │      chunks (RAG 베이스라인 전용,
 │  provenance_json, embedding BLOB     │             beliefs와 연결 없음)
 └──────────────────────────────────────┘
```

외래키 제약은 걸려 있지 않습니다. 위 화살표는 코드가 지키는 논리적 관계입니다.

### 어떤 정책이 어떤 테이블을 쓰나

| 정책 | beliefs | belief_edges | timeline_events | chunks |
|---|---|---|---|---|
| `naive_rag`, `recency_rag` | – | – | – | ✅ 쓰기/검색 |
| `structured_extract` | ✅ 추가만 | – | `ADD`만 | – |
| `belief_revision` | ✅ 추가와 상태 변경 | ✅ `SUPERSEDES` | ✅ 6가지 op 모두 | – |

### `beliefs`: 메모리 단위 (한 행 = belief 버전 하나)

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `id` | TEXT PK | `b_` + 랜덤 hex 10자리. 실행할 때마다 새로 생깁니다 |
| `type` | TEXT | `BeliefType`: static_surface / dynamic_state / workflow / gotcha / durable_rule / temporary |
| `surface` | TEXT | 무엇에 대한 belief인지 나타내는 키. 예: `billing-api.deployment_region`, `ServiceNow > Incidents > Filter`. 같은 대상이면 같은 키를 쓰는 것이 핵심입니다 |
| `claim` | TEXT | 주장 본문 |
| `closed_set_json` | TEXT(JSON list) | 화면에서 본 옵션/항목 전체 목록. "그런 버튼은 없다" 같은 전제 오류(`*-abs`) 문항에 씁니다 |
| `status` | TEXT | `active` / `superseded` / `refuted` / `forgotten`. **행은 절대 삭제하지 않습니다** |
| `confidence` | REAL 0~1 | 신뢰도. CONSOLIDATE마다 +0.05 |
| `durability` | REAL 0~1 | 오래 유지해야 하는 정도. 낮은 temporary는 IGNORE되거나 FORGET됩니다 |
| `evidence_count` | INTEGER | 이 belief를 뒷받침한 관측 수. CONSOLIDATE마다 +1 |
| `created_at_step` | INTEGER | 생성된 step(= 몇 번째 trajectory/session인지) |
| `updated_at_step` | INTEGER | 마지막으로 상태나 내용이 바뀐 step |
| `superseded_by` | TEXT | 이 belief를 대체한 새 belief의 id |
| `provenance_json` | TEXT(JSON list) | `[{source_id, step_index, url, outcome, excerpt}, ...]`. 어디서 봤는지 기록합니다 |
| `metadata_json` | TEXT(JSON) | 자유 필드 |
| `embedding` | BLOB | 임베딩 벡터(아래 3절) |

인덱스: `status`, `surface`, `type`.

### `belief_edges`: 버전 그래프

| 컬럼 | 의미 |
|---|---|
| `id` | 자동 증가 |
| `source_id` → `target_id` | `새 belief → 옛 belief` |
| `relation` | 현재 코드는 `SUPERSEDES`만 씁니다(`CONSOLIDATES`, `REFUTES`는 모델에만 정의됨) |
| `step`, `reason` | 언제, 왜(reconciler가 준 이유) |

### `timeline_events`: 감사 로그 (append-only)

| 컬럼 | 의미 |
|---|---|
| `id` | 자동 증가. 같은 step 안에서는 이 순서가 실행 순서입니다 |
| `step` | 어느 step에서 일어났나 |
| `source_id` | 어떤 trajectory/session이 일으켰나 |
| `op` | `RevisionOp` |
| `belief_id` / `target_belief_id` | 아래 표 참고 |
| `summary` | 사람이 읽는 한 줄 요약 |
| `details_json` | op별 부가 정보(옛 claim과 새 claim, 이유 등) |
| `created_at_utc` | 기록한 실제 시각 |

op별로 id 컬럼을 채우는 방식이 다릅니다. 쿼리를 짤 때 주의하세요.

| op | `belief_id` | `target_belief_id` |
|---|---|---|
| `ADD` | 새 belief | – |
| `SUPERSEDE` | **새** belief (이 경우 `ADD` 이벤트는 따로 남지 않음) | **옛** belief |
| `CONSOLIDATE` | 증거가 추가된 belief | – |
| `REFUTE`, `FORGET` | – | 상태가 바뀐 belief |
| `IGNORE` | – | – (summary에만 기록) |

### `chunks`: RAG 베이스라인의 원문 청크

| 컬럼 | 의미 |
|---|---|
| `id` | `c_` + hex |
| `source_id` | trajectory id |
| `step` | 몇 번째로 들어온 trajectory인가(recency 계산에 씀) |
| `state_index` | trajectory 안에서 몇 번째 state인가 |
| `text` | 헤더(trajectory/outcome/goal/URL/action)와 AXTree 앞부분, 최대 2400자 |
| `metadata_json` | `{"outcome": ...}` |
| `embedding` | BLOB |

## 2. 시간 여행: 과거 시점의 상태를 복원하는 방법

`beliefs.status`에는 **최종** 상태만 들어 있습니다. step *k* 시점의 상태는 `get_state_at_step(k)`(파이썬)와 `computeStatusAtStep()`(웹)이 같은 규칙으로 재구성합니다.

1. `created_at_step <= k`인 belief를 모두 `active`로 둡니다
2. `step <= k`인 `timeline_events`를 순서대로 읽으면서 `SUPERSEDE`/`REFUTE`/`FORGET`의 `target_belief_id` 상태를 바꿉니다

한계: **상태만** 복원되고, CONSOLIDATE로 덮어쓴 `claim`, `confidence`, `provenance`의 과거 값은 복원되지 않습니다(UPDATE로 덮어쓰기 때문). 옛 claim은 `details_json`에도 남지 않습니다.

## 3. 임베딩 저장과 검색 방식

- **형식**: float32 벡터를 L2 정규화한 뒤 `ndarray.tobytes()`로 BLOB에 넣습니다. 크기는 `차원 × 4바이트`입니다(qwen3-embedding:0.6b는 1024차원이라 4096바이트, 오프라인 데모는 256차원이라 1024바이트)
- **읽기**: `np.frombuffer(blob, dtype=float32)`
- **검색**: SQL로 벡터 검색을 하지 않습니다. 처음 검색할 때 임베딩 전체를 읽어 `(N, D)` 행렬로 메모리에 캐시(`_chunk_index`, `_belief_index`)한 뒤, `matrix @ query`(정규화돼 있으므로 내적 = 코사인 유사도)로 top_k를 고르고, `WHERE id IN (...)`로 해당 행만 다시 읽습니다
- **캐시 무효화**: chunks에 쓰면 `_chunk_index`, beliefs에 쓰면 `_belief_index`를 비웁니다
- **차원이 다른 벡터**: 쿼리 차원과 다른 벡터는 검색에서 조용히 빠집니다. 임베딩 모델을 바꾸면 새 DB를 쓰세요
- **임베딩 대상 텍스트**: belief는 `"[type] surface: claim"`, chunk는 `text` 전체

## 4. LLM 캐시 (`.cache/llm/llm_cache.db`)

| 테이블 | 컬럼 | 키를 만드는 방법 |
|---|---|---|
| `chat_cache` | `cache_key` TEXT PK, `response_text` TEXT | `sha256(json{base_url, model, messages, temperature, max_completion_tokens, response_format, extra_body})` |
| `embedding_cache` | `cache_key` TEXT PK, `vector` BLOB(정규화된 float32) | `sha256(json{base_url, model, text})` |

- 빈 chat 응답(`""`)은 캐시하지 않습니다
- 프롬프트가 한 글자만 바뀌어도 캐시 미스입니다. reconciler 프롬프트에는 belief id(`b_…`, 랜덤)가 들어가므로, `belief_revision`을 처음부터 다시 돌리면 reconciler 호출은 대부분 미스가 납니다. DriftSim에서는 `--resume`을 쓰세요
- 캐시를 비우려면 이 파일을 지우면 됩니다

## 5. 직접 들여다보기 (`sqlite3`)

```bash
DB=runs/demo/demo.sqlite   # scripts/demo_memory.py 결과

# 지금 활성인 belief
sqlite3 -header -column $DB "SELECT id, type, surface, claim FROM beliefs WHERE status='active'"

# 한 대상(surface)의 버전 이력
sqlite3 -header -column $DB "SELECT id, status, created_at_step, superseded_by, claim
  FROM beliefs WHERE surface='billing-api.deployment_region' ORDER BY created_at_step"

# step별 연산 로그
sqlite3 -header -column $DB "SELECT step, op, summary FROM timeline_events ORDER BY id"

# op 통계
sqlite3 $DB "SELECT op, COUNT(*) FROM timeline_events GROUP BY op"

# 출처(JSON) 펼치기
sqlite3 -header -column $DB "SELECT b.id, json_extract(p.value,'$.source_id') src,
  json_extract(p.value,'$.step_index') st, json_extract(p.value,'$.excerpt') excerpt
  FROM beliefs b, json_each(b.provenance_json) p LIMIT 10"

# 임베딩 차원 확인 (바이트 / 4)
sqlite3 $DB "SELECT length(embedding)/4 FROM beliefs LIMIT 1"
```

데모 DB에서 한 surface의 이력은 이렇게 보입니다.

```text
id            status      created_at_step  superseded_by
b_0faaa24bc3  superseded  1                b_cd222fc197
b_cd222fc197  superseded  3                b_b741cac1f8
b_b741cac1f8  active      5
```
