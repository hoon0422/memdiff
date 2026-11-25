# 코드 맵: 누가 무엇을 하는가

코드를 리뷰할 때 쓰는 지도입니다. 각 파일 안에서 어떤 클래스와 함수가 무슨 일을 맡는지, 그리고 데이터가 어떤 순서로 흐르는지 정리했습니다.

## 1. 한눈에 보는 흐름

```text
  trajectory dict (LME-V2 로그 또는 DriftSession.to_trajectory_dict())
        │
        ▼
  MemoryPolicy.insert(trajectory)          ← 정책 4개가 모두 같은 인터페이스
        │
        ├─ naive_rag / recency_rag ──► state마다 텍스트 청크 → embed_fn → BeliefStore.add_chunk()
        │
        ├─ structured_extract ───────► extractor_fn(LLM) → 후보 Belief들 → embed_fn → add_belief() (항상 추가만 함)
        │
        └─ belief_revision (MemDiff) ─► extractor_fn(LLM) → 후보 Belief마다:
                                          1) temporary이고 durability가 낮으면 → log_ignore()
                                          2) embed_fn → search_beliefs(active만) 로 이웃 검색
                                          3) reconciler_fn(LLM) → ADD / SUPERSEDE / CONSOLIDATE / REFUTE / IGNORE
                                          4) 결정에 맞는 BeliefStore 메서드 호출
                                        → 마지막에 _run_forgetting_pass(): 오래된 temporary는 FORGET

  MemoryPolicy.query(question, top_k)
        │  embed_fn(question) → search_chunks() 또는 search_beliefs() → top_k개
        ▼
  list[MemoryContextItem]  ({"type": "text", "value": "..."})
        │
        ▼
  runner가 Reader LLM에 컨텍스트와 질문을 넣음 → \boxed{답} 파싱 → metrics로 채점
        │
        ▼
  runs/<run_id>/ : *.sqlite, *_predictions.jsonl, summary/metrics.json, replay_*.json → web 뷰어
```

핵심 설계: 정책은 LLM을 직접 부르지 않습니다. `embed_fn`, `extractor_fn`, `reconciler_fn` 세 콜백을 주입받습니다. 그래서 테스트에서는 가짜 함수를 넣어 오프라인으로 돌리고, 실제 실행에서는 `llm_ops.py`가 만든 LLM 함수를 넣습니다.

## 2. 패키지별 역할

### `src/memdiff/core/`: 데이터 모델, 저장소, 인터페이스

| 파일 | 심볼 | 하는 일 |
|---|---|---|
| `models.py` | `BeliefType` | belief 종류: `static_surface`(UI 구조), `dynamic_state`(바뀌는 상태값), `workflow`(절차), `gotcha`(함정/우회법), `durable_rule`(장기 규칙), `temporary`(일회성 노이즈) |
| | `BeliefStatus` | 생명주기 상태: `active` / `superseded` / `refuted` / `forgotten`. 행을 지우지 않고 상태만 바꿉니다 |
| | `RevisionOp` | 타임라인에 기록되는 연산: `ADD`, `SUPERSEDE`, `CONSOLIDATE`, `REFUTE`, `IGNORE`, `FORGET` |
| | `Provenance` | 근거 출처: 어떤 trajectory(`source_id`)의 몇 번째 state(`step_index`)인지, 성공/실패 여부, 발췌문 |
| | `Belief` | 메모리 한 단위. `surface`(대상 키)와 `claim`(주장), 신뢰도/내구도/증거 수, 출처 목록 |
| | `BeliefEdge` | belief 사이의 방향 링크 (`새 belief --SUPERSEDES--> 옛 belief`) |
| | `TimelineEvent` | 연산 하나당 감사 로그 한 줄. 시간 여행을 재구성할 때 씁니다 |
| | `MemoryStats` | 활성/폐기 belief 수, 청크 수, 추정 토큰 수, 연산별 횟수 |
| | `ReplayBundle`, `ReplayQuestionComparison` | 웹 뷰어로 넘기는 JSON 형식. `web/src/types/replay.ts`와 1:1로 맞춰야 합니다 |
| `protocol.py` | `MemoryPolicy` | 모든 정책이 구현하는 추상 클래스: `insert()`, `query()`, `stats()` |
| `store.py` | `SCHEMA_SQL` | SQLite 테이블 정의. 자세한 내용은 [DATABASE.md](DATABASE.md) |
| | `BeliefStore.add_belief` | 새 belief 행 INSERT, `ADD` 이벤트 기록 |
| | `BeliefStore.supersede_belief` | 새 belief를 추가하고, 옛 belief를 `superseded`로 바꾸고, `SUPERSEDES` 엣지와 `SUPERSEDE` 이벤트를 기록 |
| | `BeliefStore.consolidate_belief` | 같은 belief에 증거를 추가합니다(provenance 추가, confidence +0.05, closed_set 합집합, claim 교체 가능). 행을 UPDATE합니다 |
| | `BeliefStore.refute_belief` / `forget_belief` | 상태를 `refuted` / `forgotten`으로 바꾸고 이벤트 기록 |
| | `BeliefStore.log_ignore` | belief는 건드리지 않고 `IGNORE` 이벤트만 기록 |
| | `BeliefStore.add_chunk` / `search_chunks` | RAG 베이스라인용 원문 청크 저장과 코사인 검색. `recency_decay>0`이면 오래된 청크 점수를 `exp(-decay*age)`로 깎습니다 |
| | `BeliefStore.search_beliefs` | belief 코사인 검색(`active_only` 옵션) |
| | `BeliefStore._get_chunk_index` / `_get_belief_index` | 임베딩을 `(N, D)` numpy 행렬로 메모리에 캐시합니다. 쓰기가 일어나면 캐시를 비웁니다 |
| | `BeliefStore.get_state_at_step` | 특정 step 시점의 belief 상태 스냅샷을 재구성합니다(Time Machine) |
| | `BeliefStore.stats` | `MemoryStats` 집계 |

### `src/memdiff/policies/`: 비교할 메모리 정책 4개

| 파일 | 심볼 | 하는 일 |
|---|---|---|
| `naive_rag.py` | `NaiveRagPolicy` | 베이스라인 1. trajectory의 state마다 청크 하나(목표/URL/액션/AXTree 앞부분, 최대 2400자)를 만들어 임베딩해 쌓습니다. 질의할 때는 코사인 top_k |
| `recency_rag.py` | `RecencyRagPolicy` | 베이스라인 2. 저장은 Naive와 같고, 검색할 때만 최근 것에 가중치를 줍니다 |
| `structured_extract.py` | `StructuredExtractPolicy` | 베이스라인 3. LLM으로 belief를 뽑아 그대로 추가만 합니다. 모순 처리나 망각은 없고, 검색은 `active_only=False` |
| `belief_revision.py` | `RevisionDecision` | reconciler가 돌려주는 결정(op, 대상 id, 병합 claim, 이유) |
| | `BeliefRevisionPolicy.insert` | 제안 방법(MemDiff). 위 흐름도의 1)~4) 단계를 수행합니다 |
| | `BeliefRevisionPolicy._run_forgetting_pass` | `temporary`이면서 `age >= 5`이고 `durability <= 0.5`인 활성 belief를 FORGET 처리 |
| | `BeliefRevisionPolicy.query` | 활성 belief만 검색하고 `[id | type | surface | evidence | sim]` 헤더와 출처를 붙여 반환 |
| `llm_ops.py` | `make_llm_extractor(client)` | trajectory를 요약하고(긴 UI 로그는 16개 state로 샘플링, 7000자 제한) LLM에 JSON으로 belief를 뽑게 하는 함수를 반환. 파싱 실패 시 `[]` |
| | `make_llm_reconciler(client)` | 후보 belief와 이웃(유사도 0.35 이상이거나 같은 surface)을 LLM에 보여주고 op를 고르게 하는 함수를 반환. 이웃이 없거나, 파싱에 실패하거나, 대상 id가 잘못되면 `ADD`로 폴백 |
| | `_summarize_trajectory_for_extraction` | 위 extractor가 쓰는 로그 압축기 |

### `src/memdiff/llm/`: LLM 호출

| 파일 | 심볼 | 하는 일 |
|---|---|---|
| `openai_client.py` | `CachedLLMClient.chat` | OpenAI 호환 chat 호출(Ollama 기본). 요청 전체를 해시해 `.cache/llm/llm_cache.db`에 캐시합니다. `think=False`이면 `reasoning_effort: none`을 보냅니다 |
| | `CachedLLMClient.embed` | 임베딩 호출. 텍스트별로 캐시하고, 미스만 배치로 보내고, L2 정규화합니다 |
| | `CachedLLMClient.list_ollama_models` | Ollama `/api/tags` 조회 (`check_ollama.py`에서 사용) |
| | `UsageTracker` | 호출 수, 캐시 히트 수, 토큰 수 집계 |

### `src/memdiff/config.py`

| 심볼 | 하는 일 |
|---|---|
| `Settings` | `.env`를 읽습니다: LLM 엔드포인트와 키, 모델 이름 3개(chat/embedding/judge), 타임아웃, 데이터와 캐시 경로 |
| `get_settings()` | `Settings()` 생성 |

### `src/memdiff/benchmarks/drift_sim/`: Track 2 (합성 벤치마크)

| 파일 | 심볼 | 하는 일 |
|---|---|---|
| `simulator.py` | `DriftSimConfig` | 시드, 세션 수, 엔티티 수, 모순 확률, 세션당 노이즈 수, 규칙 수 |
| | `generate_episode(cfg)` | 결정적으로 세션들을 만듭니다. 1~N번 세션에서 규칙을 하나씩 소개하고, 1번 세션에서 상태를 초기화하고, 이후 세션은 확률적으로 값을 바꾸고, `[TEMP LOG]` 노이즈를 넣습니다. 질문(현재 값, 규칙)과 정답, stale 오답 목록도 함께 생성합니다 |
| | `DriftSession.to_trajectory_dict()` | 세션을 LME-V2와 같은 trajectory dict 모양으로 변환 |
| `runner.py` | `build_default_policy(name, store, client)` | 정책 이름으로 인스턴스를 만듭니다 |
| | `evaluate_policy_on_episode` | 정책 하나에 대해 ingest → 질문마다 query → Reader LLM → `\boxed{}` 파싱 → 채점 |
| | `score_drift_prediction` | `(정답 여부, stale 오답 여부)` 반환 |
| | `aggregate_drift_predictions` | `stale_belief_rate`, `durable_rule_preservation` 등 집계 |
| | `run_drift_sim_benchmark` | 여러 정책을 실행하고 `runs/<id>/`에 sqlite, predictions, summary, replay를 저장. `--resume` 지원 |
| | `main()` | CLI (`scripts/run_drift_sim.py`) |

### `src/memdiff/benchmarks/lmev2/`: Track 1 (실제 로그 벤치마크)

| 파일 | 심볼 | 하는 일 |
|---|---|---|
| `fetcher.py` | `fetch_lmev2_small` / `main` | HuggingFace에서 질문, haystack, 질문 스크린샷을 받고, `trajectories.jsonl`은 필요한 id만 스트리밍으로 걸러 저장 |
| `loader.py` | `load_questions`, `load_haystack`, `load_trajectories` | `data/longmemeval-v2/` 아래 파일 로드. haystack은 "질문 id → 순서가 있는 trajectory id 목록" |
| `metrics.py` | `extract_boxed_answer`, `is_unknown` | 응답에서 `\boxed{}` 추출, UNKNOWN 판정 |
| | `norm_phrase_set_match`, `mc_choice_match` 등 | upstream과 같은 결정적 채점 함수 |
| | `llm_abstention_checker`, `llm_gotchas_checker` | `*-abs`와 gotcha 문항을 LLM judge로 채점 |
| | `eval_from_spec(spec, pred, answer)` | 질문의 `eval_function` 문자열을 보고 알맞은 채점 함수로 보냅니다 |
| | `aggregate_metrics` | 카테고리별 정답/오답/UNKNOWN 비율 |
| `runner.py` | `DOMAIN_SYSTEM_PROMPTS` | Reader 시스템 프롬프트(upstream과 같아야 함) |
| | `select_balanced_questions` | 질문 유형별로 N개씩 샘플링 |
| | `_build_reader_messages` | 컨텍스트, 질문, (선택) 이미지로 Reader 메시지 구성 |
| | `run_lmev2_policy` | haystack이 같은 질문끼리 묶어 그룹마다 새 store에 ingest한 뒤 채점하고 결과를 저장 |
| | `main()` | CLI (`scripts/run_lmev2.py`) |

### `src/memdiff/export/`

| 파일 | 심볼 | 하는 일 |
|---|---|---|
| `replay.py` | `build_replay_bundle(store, ...)` | store의 beliefs, edges, events, stats를 `ReplayBundle`로 묶음 |
| | `export_replay_json(bundle, path)` | JSON 파일로 저장 |

### `scripts/`: 실행 진입점

| 파일 | 하는 일 |
|---|---|
| `demo_memory.py` | **가장 먼저 해볼 것.** 작은 DriftSim 에피소드로 insert/query를 보여줍니다. 기본은 LLM 없이 오프라인, `--live`는 실제 LLM |
| `check_ollama.py` | Ollama 연결과 모델 설치 여부 확인 |
| `fetch_lmev2_small.py` | LME-V2 데이터 다운로드 (`lmev2/fetcher.py:main`) |
| `run_drift_sim.py` | Track 2 실행 (`drift_sim/runner.py:main`) |
| `run_lmev2.py` | Track 1 실행 (`lmev2/runner.py:main`) |

### `web/`: Memory Time Machine 뷰어

| 파일 | 하는 일 |
|---|---|
| `src/types/replay.ts` | `ReplayBundle` TypeScript 타입(파이썬 `models.py`와 동기화) |
| `src/App.tsx` | 슬라이더로 step을 고르면 `computeStatusAtStep()`이 그 시점의 belief 상태를 계산해 보여줍니다. 오른쪽 위 "Load replay JSON"으로 `runs/.../replay_*.json`을 불러옵니다 |
| `src/data/sampleReplay.ts` | 처음 열면 보이는 샘플 데이터 |

## 3. 리뷰할 때 먼저 볼 곳

1. `core/models.py`: 용어 정의(5분)
2. `policies/belief_revision.py`의 `insert()`: 이 프로젝트의 핵심 로직
3. `core/store.py`의 `supersede_belief()`와 `get_state_at_step()`: "지우지 않고 버전 관리"가 실제로 어떻게 구현됐는지
4. `policies/llm_ops.py`의 두 프롬프트: 실제 품질은 여기서 결정됩니다
5. `tests/test_store.py`: 위 동작을 짧게 실행해 보이는 예제
