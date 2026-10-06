# IMPLEMENT.md — Cold-Start-Aware, Explainable E-Commerce Recommendation

**Course:** CSD358 IR Hackathon (36 h) | **Track:** T3, Recommender systems and personalization
**Audience for this file:** the team *and* the AI coding agent (Claude Code / Cursor / Codex / Copilot) that will write the code.

> **One-line pitch.** Products are *documents*, a shopper's history is a *query*. The system works from product text alone when an item or user is new, sharpens as purchase behaviour accumulates (Amazon-style item-to-item signal), and explains every recommendation with evidence whose **faithfulness we measure**.

---

## 0. Ground rules (read first, apply always)

These come straight from the assignment and are graded.

1. **IR must be a real, inspectable part of the system.** The inverted index, SMART weighting, cosine scoring, heap top-K, champion lists, cluster pruning, zones, parametric filters, static quality `g(d)`/net score and Jaccard are **implemented by us** (numpy/scipy are fine as array engines). Do **not** replace them with a black-box library call (e.g. `TfidfVectorizer` + `cosine_similarity` as the main pipeline). sklearn may be used **only in tests** to cross-check our numbers.
2. **Nothing hard-coded, nothing faked.** All demo output must come from running the code on real data. A faked/hard-coded demo scores 0 for "working system".
3. **No UI work.** CLI + a notebook only. Frontend is not graded.
4. **No data leakage.** Every statistic used at recommendation time (popularity, rating quality, recency, co-purchase/Jaccard) must be computed **from the training snapshot only**. See §5.
5. **Reproducible.** Fixed seeds, config-driven, one `make` target per stage, README reproduces the demo from scratch.
6. **Declare AI use.** Maintain `docs/ai_use_log.md` (§14) as you go.
7. **Credit the dataset, anonymise users** (integer ids only; never commit raw user ids).
8. **Be honest in results.** Report what the numbers say, including when a novelty idea does *not* help. Hypotheses in this document are hypotheses, not claims.
9. **Scope discipline.** Must-ship first, then two novelty items, then stretch. Cut list: LambdaMART, DPP, LLM reranker, any UI.

---

## 1. What we are building

A hybrid product recommender with five layers:

| Layer | What it does | Lecture concept |
|---|---|---|
| Content retrieval | Product = document (title / body / category zones); user = query built from history; cosine ranking | Vector space model, lnc.ltc, zone indexes |
| Behaviour signal | Item↔item reviewer-set overlap (complements) | Jaccard coefficient |
| Static quality | Popularity, Bayesian rating quality, recency | `g(d)`, net score |
| Efficiency | Heap top-K, champion lists, cluster pruning, parametric filters | Efficient cosine scoring, pruning |
| Trust | Term-level + behaviour-level explanations, counterfactually tested | (novelty) |

**Novelty items we ship (pick two; this plan commits to A + B):**
- **A. Complement vs. substitute blend** weighted by per-item interaction count, with a quality-vs-interactions (cold-start) curve.
- **B. Faithfulness-tested explanations** ("because you bought X", top contributing terms, co-purchase reason; remove X and check the rec drops).
- **C. (stretch) MMR + variant collapsing** — only after A, B and all experiments are done (§11).

---

## 2. Tech stack and dependencies

Python ≥ 3.10. CPU only; no GPU needed.

```
numpy, scipy, pandas, pyarrow, pyyaml, tqdm, matplotlib, typer, huggingface_hub
nltk            # PorterStemmer only (no corpus downloads)
scikit-learn    # ENGLISH_STOP_WORDS list + tests-only cross-checks
pytest
# optional / stretch
datasketch              # MinHash LSH for shingle near-duplicates (stretch C)
sentence-transformers   # dense comparison channel (optional)
```

For the report, every library must be explained **in IR terms** (e.g. "scipy CSR = our postings storage; datasketch MinHash = approximate Jaccard over shingle sets"). Keep this list in `docs/ir_mapping.md`.

---

## 3. Repository layout

```
.
├── README.md                  # setup, data source, how to run, what works / what's planned
├── IMPLEMENT.md               # this file
├── AGENT_PROMPT.md            # prompts for the coding agent
├── Makefile                   # one target per stage
├── requirements.txt
├── .gitignore                 # data/, results/*.parquet, *.npz, __pycache__, .venv
├── configs/
│   └── default.yaml
├── data/                      # (gitignored) raw/ and processed/
├── src/ecrec/
│   ├── __init__.py
│   ├── config.py              # load yaml -> dataclasses
│   ├── cli.py                 # typer app: all commands in §12
│   ├── data/
│   │   ├── download.py        # HF hub download of raw jsonl
│   │   ├── prepare.py         # clean, join, sample, anonymise -> parquet
│   │   └── split.py           # temporal splits + stat snapshots
│   ├── text/
│   │   ├── tokenizer.py      # tokenise / normalise / stem / stopwords / units / product codes
│   │   └── zones.py           # build zone texts: title, body, meta
│   ├── index/
│   │   ├── inverted_index.py  # dictionary + postings, df, idf, doc norms
│   │   ├── weighting.py       # SMART scheme parser: lnc, ltc, ltn, bm25
│   │   ├── scoring.py         # term-at-a-time cosine, zone combination
│   │   ├── topk.py            # heap-based top-K (+ argpartition variant)
│   │   ├── champion.py        # per-term and per-category/interest champion lists
│   │   ├── cluster_pruning.py # leaders & followers
│   │   └── parametric.py      # category/brand/price filters + tiny query parser
│   ├── profile/
│   │   └── user_profile.py    # history -> query vector (weighted bag, centroid)
│   ├── cf/
│   │   └── jaccard.py         # item-item Jaccard neighbour lists
│   ├── quality/
│   │   └── static_quality.py  # popularity, rating quality, recency, net score
│   ├── recommend/
│   │   ├── hybrid.py          # content + CF blend with per-item lambda, net score
│   │   └── baselines.py       # popularity, centroid tf-idf, content-only, CF-only, BM25
│   ├── explain/
│   │   ├── explain.py         # term-level + behaviour-level explanations
│   │   └── faithfulness.py    # counterfactual tests
│   ├── rerank/                # STRETCH C
│   │   ├── shingles.py
│   │   └── mmr.py
│   └── eval/
│       ├── metrics.py         # Hit/P/Recall/NDCG@k, coverage, ILD
│       ├── harness.py         # protocol runner
│       └── experiments.py     # ablation, pruning speed, cold-start, faithfulness
├── scripts/                   # thin wrappers calling cli commands (optional)
├── notebooks/demo.ipynb       # live demo notebook (CLI is equally fine)
├── tests/
├── results/                   # csv tables + json (small ones committed)
├── figures/                   # png graphs for the report
└── docs/
    ├── ir_mapping.md          # lecture concept -> file:function -> why (feeds report §2)
    ├── decisions.md           # design decisions + rationale
    ├── ai_use_log.md          # AI-use declaration (feeds report)
    ├── work_division.md
    ├── hand_judged.csv        # team-labelled sanity check
    └── report_notes.md        # figure/table list mapped to report sections
```

---

## 4. Configuration (`configs/default.yaml`)

All tunables live here; the code reads them, never hard-codes them.

```yaml
seed: 42

data:
  hf_repo: McAuley-Lab/Amazon-Reviews-2023
  category: Musical_Instruments        # swap to Cell_Phones_and_Accessories etc. if bandwidth allows
  meta_file:   raw/meta_categories/meta_{category}.jsonl
  review_file: raw/review_categories/{category}.jsonl
  max_items: 40000          # catalogue size after sampling
  min_item_reviews: 3       # to be eligible for the sampled catalogue
  min_user_interactions: 3  # need train>=1, val=1, test=1
  verified_only: true       # reviews as purchase proxy; set false if too little data
  min_text_tokens: 5        # drop items with ~no text

split:
  mode: leave_last_out      # per brief. optional robustness check: global_time
  eval_users: 3000          # sampled with seed for test; 1000 for validation/tuning
  tune_users: 1000

text:
  stem: true
  stopwords: true
  keep_codes: true          # model numbers, units (usb-c, 10000mah, 256gb)
  zones: {title: 0.5, body: 0.3, meta: 0.2}   # g_z, tuned on validation

weighting:
  doc: lnc                  # SMART
  query: ltc
  query_max_terms: 50       # index elimination: keep top-m query terms by weight

profile:
  mode: weighted_bag        # or centroid (baseline)
  half_life: 3              # history-position decay: w = 0.5 ** (age_rank / half_life)

cf:
  max_user_degree: 200      # drop bot-like users from co-occurrence
  min_co: 2
  neighbours: 50
  shrinkage: 3              # J' = co / (|Ui|+|Uj|-co + shrinkage)

hybrid:
  kappa: 5                  # lambda(n) = n / (n + kappa)
  alpha: 0.05               # popularity
  beta: 0.05                # rating quality
  gamma: 0.02               # recency
  recency_tau_days: 365
  rating_prior_m: 10        # Bayesian average strength

pruning:
  champion_r: 300
  leaders: sqrt_n
  cluster_b: 2              # leaders probed

eval:
  ks: [5, 10, 20]
  faithfulness_k: 10
  faithfulness_users: 500
  random_control_seeds: 5
```

---

## 5. Data pipeline (Phase 1)

### 5.1 Source
**Amazon Reviews 2023** (Hou et al. 2024, McAuley Lab) from Hugging Face: `McAuley-Lab/Amazon-Reviews-2023`. Use `huggingface_hub.hf_hub_download(repo_type="dataset", ...)` for the two raw files of one category. **Verify exact file names and sizes on the dataset card before downloading**, and pick a category you can download in reasonable time (Electronics and Home & Kitchen are very large; a mid-size category is enough — "a few tens of thousands of products is enough"). Credit the dataset in README and report.

Fields used:
- **meta:** `parent_asin, title, features (list), description (list), categories (list), main_category, store, price, average_rating, rating_number`
- **reviews:** `user_id, parent_asin, rating, timestamp (milliseconds), verified_purchase, helpful_vote`

Join on `parent_asin` (not `asin`).

### 5.2 Cleaning and sampling (`data/prepare.py`)
1. Stream reviews; keep `verified_purchase` if configured; **dedupe** `(user, item)` keeping the earliest.
2. Stream meta; drop items with no title or fewer than `min_text_tokens` tokens across title+features+description.
3. Eligible catalogue = items with ≥ `min_item_reviews` reviews and usable text. Random-sample `max_items` (seeded). **Do not select by popularity** — that would erase the cold-start regime we are studying.
4. Keep interactions on catalogue items; keep users with ≥ `min_user_interactions`.
5. Anonymise: map `user_id → user_idx`, `parent_asin → item_idx` (0..N-1). Discard raw user ids after mapping (never write them to committed files).
6. Brand: `store` field, falling back to a `details["Brand"]`-style value if present.
7. Write `data/processed/items.parquet`, `data/processed/interactions.parquet`, plus `data/processed/stats.json` (counts, sparsity, timestamp range).

**`items.parquet` schema:** `item_idx, parent_asin, title, features, description, categories, main_category, leaf_category, brand, price (nullable)`.
**`interactions.parquet` schema:** `user_idx, item_idx, rating, ts, helpful_vote, split`.

### 5.3 Splits (`data/split.py`) — protocol from the brief
Per user, sort by `(ts, row order)`:
- `test`  = last interaction (the target)
- `val`   = second-to-last (used only for tuning)
- `train` = everything before

**Two statistic snapshots** (this is the anti-leakage rule):
- `S_val`  = statistics from `train` only → used when evaluating on `val` targets (tuning).
- `S_test` = statistics from `train + val` → used when evaluating on `test` targets.

User history at evaluation: for val, history = train items; for test, history = train + val items. Candidate set = **all** catalogue items minus the user's history items. Never restrict candidates to items seen in training — items with zero training interactions must be rankable; that *is* the cold-start case.

Everything that depends on interactions is computed from the snapshot only: popularity, rating quality (**do not use meta `average_rating`/`rating_number`** — computed on all reviews, so it leaks the future), recency, Jaccard neighbours.

Item `n_i` = number of interactions in the snapshot (used for λ and for cold-start buckets).

**Known limitation to state:** per-user leave-last-out still lets other users' later interactions appear in training. Optionally implement `split.mode: global_time` (cutoff = a timestamp quantile; train = before cutoff; target = a user's first interaction after cutoff) and re-run the headline comparison as a robustness check.

**Acceptance (Phase 1):** `make data` produces both parquet files; a test asserts no `test`/`val` row is in the train snapshot, each user has exactly one test and one val row, and no `(user,item)` duplicates.

---

## 6. IR core (Phases 2–4)

### 6.1 Text processing — `text/tokenizer.py` (lecture: term vocabulary)
Pipeline: lowercase (case folding) → regex tokenisation that **keeps product codes and units** → normalise units (`10000 mAh`, `10000mah` → `10000mah`; `256 GB` → `256gb`; `usb-c`/`usb c` → `usbc`) → stop-word removal (sklearn `ENGLISH_STOP_WORDS`) → Porter stemming (`nltk.PorterStemmer`).
- Provide `tokenize(text, stem=True, stop=True, codes=True) -> list[str]` — flags exist so the ablation can switch each step off.
- Keep a `stem → most frequent surface form` map; explanations display surface forms, not stems.
- Concatenate list-valued fields (`features`, `description`, `categories`) with spaces.

### 6.2 Zones — `text/zones.py` (lecture: zone indexes)
Three zones per product: `title`, `body` (features + description), `meta` (categories + brand). Build **one inverted index per zone** (same vocabulary ids across zones so a user's query vector is shared). Document score = `Σ_z g_z · cos_z(q, d)`, with `g_z` from config (tuned on validation).

### 6.3 Inverted index — `index/inverted_index.py` (lecture: Boolean retrieval + postings)
- Dictionary: `term -> term_id`, `df[term_id]`, `idf = log10(N/df)`.
- Postings: for each term a pair of arrays `(doc_ids ascending, tf)`. Store as CSC of the doc×term matrix (column = postings list) wrapped in a class exposing `postings(term_id)`, `df(term_id)`, `idf(term_id)`, `doc_norm(doc_id)`.
- Support Boolean AND/OR/NOT over postings (sorted-array merge; AND processes terms in increasing `df` order) — used by the parametric filter + keyword search and for the video ("show postings").
- Persist with `scipy.sparse.save_npz` + `pickle`/json for vocab.

### 6.4 SMART weighting — `index/weighting.py` (lecture: tf-idf, lnc.ltc)
Parse scheme strings. Implement at minimum:
- **Documents `lnc`:** `w = 1 + log10(tf)`, no idf, cosine-normalise the doc vector.
- **Queries `ltc`:** `w = (1 + log10(tf)) · idf`, cosine-normalise.
- Extra schemes for ablation: `ltn`/`lnn` (no length norm), `lnc.lnc` (no idf).
- **BM25** (extra points) over the *same* index: `k1=1.2, b=0.75` (comparison channel only).

### 6.5 Scoring — `index/scoring.py` + `index/topk.py` (lecture: efficient cosine scoring)
Term-at-a-time with an accumulator array:

```python
def score_taat(index, q: dict[int, float]) -> np.ndarray:
    scores = np.zeros(index.n_docs, dtype=np.float32)
    for t, wq in q.items():                  # process high-idf terms first
        docs, wd = index.postings(t)         # wd already length-normalised
        scores[docs] += wq * wd              # doc ids unique within a list
    return scores
```

Heap top-K (`topk.py`): `top_k_heap(scores, k, exclude=None, candidates=None)` using a size-K min-heap via `heapq`; ties broken by smaller doc id (deterministic). Also expose an `np.argpartition` variant and a test that both agree. Support `candidates` (restricted scoring) and `exclude` (history items).

**Index elimination:** truncate the query to the top `query_max_terms` terms by weight (and optionally only consider docs containing ≥1 high-idf term).

### 6.6 User profile as query — `profile/user_profile.py`
Two builders (both return a sparse term-weight dict per zone):
- **`centroid` (baseline):** mean of the unit-length tf-idf vectors of history items (the "single-centroid tf-idf profile").
- **`weighted_bag` (main):** `q_t = Σ_i w_i · (1 + log10 tf_{i,t}) · idf_t`, where `w_i = 0.5 ** (age_rank_i / half_life)` (most recent item has age 0), optionally × rating weight; then cosine-normalise (`ltc`) and truncate to `query_max_terms`.

Because `weighted_bag` is linear in per-item contributions before normalisation, a candidate's content score decomposes **exactly** into per-history-item contributions — this powers the explanations (§8). Keep the decomposition code path tested (`sum of contributions == score`).

Item-to-item "similar items": the same machinery with a single item as the query.

### 6.7 Static quality and net score — `quality/static_quality.py` (lecture: `g(d)`, net score)
All from the snapshot, all normalised to [0, 1]:
- `pop(d) = log(1 + n_d) / log(1 + max_n)`
- `rq(d)` = Bayesian average rating `(m·μ + Σ r) / (m + n)` → `(x − 1) / 4`; `μ` = global mean rating in snapshot; `m` = `rating_prior_m`. Items with no interactions get `rq = (μ − 1)/4` (neutral).
- `rec(d) = exp(−(T_cut − last_ts_d) / τ)`; items with no interactions get 0. `T_cut` = the snapshot's latest timestamp.

`net(d) = rel(d) + α·pop(d) + β·rq(d) + γ·rec(d)`.
**Tune** `(α, β, γ, g_title, g_body, g_meta, κ, half_life)` on the **validation** split (NDCG@10) with seeded random search (~60 trials on `tune_users`). Log all trials to `results/tuning.csv`. Never tune on test.

### 6.8 Jaccard co-purchase — `cf/jaccard.py` (lecture: Jaccard coefficient)
Reviewer-set overlap between items, from the snapshot:

```
B   = binary item×user CSR (users with degree > max_user_degree removed)
co  = B @ B.T computed in row chunks (e.g. 2000 items at a time)
J_ij = co_ij / (|U_i| + |U_j| - co_ij + shrinkage)       # keep co_ij >= min_co
store top `neighbours` per item as (indices, scores) arrays
```
Never materialise a dense item×item matrix. This is the Amazon item-to-item idea (Linden et al. 2003), done offline, with fast lookup at query time. Brief says content finds substitutes and co-purchase finds complements — keep these two channels separate until the blend (§7).

### 6.9 Champion lists — `index/champion.py` (lecture: champion lists)
- **Per-term champion lists:** top-`r` docs by weight for each term (`r ≥ K`). Candidate set = union over the query's terms; exact cosine on candidates; fall back to full scoring if `|candidates| < K`.
- **Per-category static lists:** top-`r` items per leaf category by `g(d)` (also used as a **no-history fallback** for brand-new users).
- **Per-interest lists:** for a user, candidates from the champion list of each of their top history categories.

### 6.10 Cluster pruning — `index/cluster_pruning.py` (lecture: leaders and followers)
- Pick `√N` random leaders (seeded). Each doc follows its nearest leader (sparse matrix product `D @ Lᵀ`, `argmax` per row, chunked). Optionally attach each doc to its 2 nearest leaders.
- Query: score against leaders, probe the top `b` leaders, take the union of their followers, run exact cosine on those followers only.
- Expose `b` as a parameter. The experiment (§10.3) reports **latency vs. recall-of-brute-force-top-K** for `b ∈ {1, 2, 3}` and champion `r ∈ {50, 100, 300, 1000}`.

### 6.11 Parametric indexes + query parser — `index/parametric.py` (lecture: parametric/zone indexes, query parsers)
- Parametric fields: `leaf_category`, `main_category`, `brand` (dict → sorted doc-id arrays), `price` (sorted array + `bisect` range).
- Filters AND-ed with cosine ranking: compute the allowed doc-id set from the parametric indexes, then restrict scoring/top-K to it.
- Tiny parser for the `search` command: `"waterproof bluetooth speaker brand:Anker price<50 category:\"Portable Speakers\""` → free-text terms + filters.

---

## 7. Hybrid recommender — `recommend/hybrid.py`

```
recommend(user_history, k, filters=None, mode="hybrid") -> list[Rec]

1. content[c]  = Σ_z g_z · cos_z(q_user, d_c)                    # §6.2–6.5; full | champion | cluster-pruned
2. cf[c]       = Σ_i w_i · J(i, c)  for history items i           # sparse accumulate over neighbour lists
3. normalise content and cf per query (divide by max over candidates)
4. λ_c         = n_c / (n_c + κ)         # n_c = interactions of candidate c in the snapshot
5. rel_c       = (1 − λ_c) · content_c + λ_c · cf_c
6. net_c       = rel_c + α·pop_c + β·rq_c + γ·rec_c
7. apply parametric filters; exclude history items
8. heap top-K → (optional) variant collapse + MMR (stretch C)
```

- A brand-new item (`n_c = 0`) is scored by **content only**; a heavily-reviewed item leans on **behaviour**. That is Novelty A.
- Return per rec: item_idx, net score, rel, content, cf, λ, channel label (§8), and the score breakdown (for the `inspect-score` command).
- A no-history user returns the per-category champion list ordered by `g(d)`.
- `mode` ∈ `hybrid | content_only | cf_only | popularity | centroid | bm25` so every baseline and ablation goes through the same harness.

**Channel label (substitute vs. complement):** for each rec, find the history item contributing most; label `substitute` if the rec shares its `leaf_category`, else `complement`. This is a *measured proxy*; say so in the report.

---

## 8. Explanations (Novelty B) — `explain/explain.py`

For each recommendation produce:

```json
{
  "item": "<title>", "rank": 1, "net": 0.43,
  "channel": "substitute | complement | mixed",
  "because_you_bought": {"item_idx": 123, "title": "..."},
  "content_evidence": {"cosine": 0.31, "top_terms": [["waterproof", 0.082], ["bluetooth", 0.061]]},
  "behaviour_evidence": {"jaccard": 0.07, "co_reviewers": 14},
  "static": {"pop": 0.4, "rating_quality": 0.8, "recency": 0.3}
}
```

- **Cited history item** `x* = argmax_i w_i·[(1−λ)·content(i, c) + λ·J(i, c)]`.
- **Term contributions** = per-term products `q_t · d_t` for the candidate (summed over zones with `g_z`), sorted; show surface forms; top 3–5.
- **Behaviour reason** only when the CF term materially drove the score (`λ·cf` share above a threshold, e.g. 30% of `rel`): "N shoppers who reviewed X also reviewed this (Jaccard = 0.07)".
- Print as a short natural-language sentence **and** keep the JSON.

### 8.1 Faithfulness tests — `explain/faithfulness.py`
Use `faithfulness_users` users with history length ≥ 2 and their top-`faithfulness_k` recs.

1. **Item-level counterfactual (the brief's test).** Remove the cited item `x*` from the history, recompute recommendations. Record: (a) did rec `c` leave the top-K? (b) rank shift, (c) score drop.
2. **Random-item control.** Remove a *random other* history item instead (`random_control_seeds` repetitions). A faithful explanation must show a clearly larger drop rate than the control.
3. **Term-level deletion test.** Zero out the cited top-`t` terms (t = 3) in the query vector and measure the score drop for `c`, versus deleting `t` random terms from the same query. Report both.
4. **Report:** `drop@K(cited)`, `drop@K(random)`, mean rank shift, mean score delta — as a table plus a bar chart, split by channel (substitute/complement) and by item interaction bucket.

Write the results to `results/faithfulness.csv`. Limitation to state: only single-history-item removal is tested, not combinations.

---

## 9. Baselines

| Name | Definition |
|---|---|
| `popularity` | rank by snapshot interaction count (excluding history) |
| `centroid` | single-centroid tf-idf profile, plain cosine, no zones, no net score |
| `content_only` | full content pipeline, no CF, no net score |
| `cf_only` | Jaccard channel only |
| `bm25` | BM25 over the same index (comparison channel, extra points) |
| `dense` *(optional)* | `all-MiniLM-L6-v2` sentence embeddings of title+description, mean of history embeddings, cosine |

---

## 10. Evaluation (Phase 7)

### 10.1 Protocol
- Tuning on `val` targets with `S_val`; final numbers on `test` targets with `S_test` (§5.3).
- Eval users: `eval_users` sampled with the seed (test), `tune_users` (val). All methods use the **same** users.
- Candidates = whole catalogue minus history. Single target per user → `Hit@k == Recall@k`, `P@k = Hit@k / k`; state this in the report.
- Metrics (`eval/metrics.py`): Hit@k, P@k, Recall@k, NDCG@k for k ∈ {5, 10, 20}; **catalogue coverage** (share of items appearing in any user's top-K); **intra-list diversity** (1 − mean pairwise cosine among a list's items).
- Bootstrap 95% CIs over users for the headline table.

### 10.2 Main comparison table
`popularity | centroid | content_only | cf_only | bm25 | hybrid` × all metrics → `results/main_table.csv` + a bar chart.

### 10.3 Experiments (each writes a CSV and a figure)
1. **Ablation** (`results/ablation.csv`): full system, then each component off: zones → single zone; no stemming; no stop words; no idf (`lnc.lnc`); no length norm; cosine only (no net score); each of α/β/γ → 0; no CF (κ→∞); no content; `centroid` instead of `weighted_bag`; no query truncation.
2. **Pruning vs brute force** (`results/pruning.csv`): latency (ms/query, median and p95, many repetitions with `time.perf_counter`) and recall of the brute-force top-K, plus Hit@10, for brute force, champion lists (`r` grid), cluster pruning (`b` grid). Plot the speed–recall trade-off.
3. **Cold-start curve** (`results/coldstart.csv`): bucket test targets by their snapshot interaction count `n ∈ {0, 1–2, 3–5, 6–10, 11–50, 51+}`; plot Hit@10 (y) against bucket (x) for popularity, content_only, cf_only, hybrid with fixed λ, hybrid with adaptive λ. Report bucket sizes. (Hypothesis: content wins at low `n`, behaviour at high `n`, the adaptive blend tracks the upper envelope — **report whatever happens**.)
4. **Substitute vs complement** (`results/channels.csv`): share of recs in each channel, cross-category rate per channel, and Hit@10 per channel.
5. **Faithfulness** (§8.1).
6. **Hand-judged sanity check:** the CLI writes ~15 users' history + top-5 recs + explanations to `docs/hand_judged.csv` (empty `judgement` column). The team labels each rec 0/1/2 (irrelevant / plausible / good); a script computes judged P@5 for hybrid vs centroid. This also feeds the "real queries" part of the video.

### 10.4 Honest limitations to state
Reviews are a proxy for purchases; offline metrics only approximate shopper satisfaction; product text quality varies widely; the counterfactual test covers single history items; per-user leave-last-out leaks some global time information; English only; popularity bias in Amazon data.

---

## 11. Stretch items (only after Phases 0–8 are green)

- **C. MMR + variant collapsing.** Shingle titles (word 3-grams), exact Jaccard ≥ 0.8 (or MinHash-LSH via `datasketch` at index time, verified exactly) → group variants (union-find), keep the highest-`g(d)` member. Then MMR over the top-100: `argmax λ_mmr·net(c) − (1−λ_mmr)·max_{s∈S} cos(c, s)`, `λ_mmr = 0.7`. Report coverage and ILD vs Hit@10 for `λ_mmr ∈ {1.0, 0.9, 0.7, 0.5}`.
- **Dense retrieval channel** and/or **BM25 fusion** (RRF) as a comparison.
- **Target-attention over history** vs the centroid baseline, tested with the same counterfactual check.

---

## 12. CLI (`python -m ecrec.cli ...`)

Every command is deterministic given the config and works from cached artefacts.

| Command | Purpose |
|---|---|
| `download` | fetch raw files for the configured category |
| `prepare` | clean/sample/anonymise/split → parquet |
| `build-index` | tokenise, build zone indexes, Jaccard neighbours, champion lists, leaders/followers, static stats |
| `search "<query> brand:X price<50"` | free-text + parametric product search, shows ranked results |
| `recommend --user <user_idx> --k 10 [--explain] [--mode hybrid]` | recommendations (+ explanations) |
| `inspect-term <term>` | dictionary entry: df, idf, first N postings (doc_id, tf, weight) |
| `inspect-item <item_idx>` | tokens, per-zone weight vector (top terms), norm |
| `inspect-score --user <u> --item <i>` | full breakdown: per-zone cosine, term contributions, CF, λ, g(d), net |
| `tune` | random search on validation → writes `results/tuning.csv` and best config |
| `eval` | main comparison table on test |
| `experiments --which ablation|pruning|coldstart|channels|faithfulness|all` | run experiments, write CSVs + figures |
| `hand-judge` | emit the sheet for manual judgement |

`inspect-*` commands exist so the demo video can **show intermediate output** (postings, weights, scores) rather than just final screens.

`Makefile` targets: `setup`, `data`, `index`, `tune`, `eval`, `experiments`, `demo`, `test`, `all`.

---

## 13. Testing (`tests/`) — must pass before each commit

- **Weighting:** a hand-computed lnc.ltc example (small 3-doc corpus) matches our vectors and cosine scores to 1e-6.
- **Index:** postings are sorted, `df` matches counts, Boolean AND over postings equals a Python set intersection.
- **Top-K:** heap result == full `argsort` result including tie-breaking; excludes history items.
- **Champion lists:** candidates ⊆ catalogue; with `r = N` results equal brute force.
- **Cluster pruning:** probing all leaders equals brute force.
- **Jaccard:** matches a naive Python-set computation on a toy interaction table; shrinkage and `min_co` respected.
- **Decomposition:** per-history-item contributions sum to the content score.
- **No leakage:** snapshot statistics contain no val/test rows; each user has one val + one test; `S_val ⊂ S_test` relationship holds.
- **Determinism:** same seed → identical top-K.
- **Metrics:** Hit/NDCG on hand-made rankings.

Small fixtures in `tests/fixtures/` are for tests **only**; never use them in the demo.

---

## 14. Documentation artefacts (maintain continuously)

- **`README.md`:** setup, dataset source/credit, exact commands to reproduce each stage and the demo, runtime expectations, **what works / what's planned**, limitations.
- **`docs/ir_mapping.md`:** table `lecture concept → file:function → why we chose it → library/tool used (explained in IR terms)`. This becomes report §2.
- **`docs/decisions.md`:** dated design decisions with rationale.
- **`docs/ai_use_log.md`:** `date | tool/model | what it was used for | files touched | human verification`. Declare coding agents, LLMs, pretrained models.
- **`docs/work_division.md`:** who owned which component (information only; no percentages).
- **`docs/report_notes.md`:** list of figures/tables each report section needs (§16).

---

## 15. Phased plan with acceptance criteria

Aligned to the 36 h plan. Commit at the end of every phase; do not start a phase until the previous one's acceptance criteria pass.

| Phase | Hours | Deliverable | Acceptance |
|---|---|---|---|
| **0 Scaffold** | 0–1 | repo layout, config, requirements, Makefile, CLI skeleton, empty tests | `make test` runs; `python -m ecrec.cli --help` works |
| **1 Data** | 0–4 | download/prepare/split, stats | §5.3 tests pass; `stats.json` shows N items, U users, #interactions, sparsity |
| **2 Text + index** | 2–8 | tokenizer, zones, inverted index, SMART weights, `inspect-term`, `inspect-item` | weighting + index tests pass; `inspect-term` prints real postings |
| **3 Retrieval + eval harness** | 4–12 | TAAT cosine, heap top-K, user profiles, popularity/centroid/content_only baselines, metrics, harness | first Hit@k/NDCG table for baselines on val |
| **4 Quality + CF + efficiency** | 12–20 | static quality, net score, Jaccard neighbours, hybrid blend, champion lists, cluster pruning, parametric filters + parser, `tune` | hybrid ≥ or ≈ baselines on val *or the gap is documented*; pruning tests pass |
| **5 Explanations** | 12–20 | explanation JSON + sentence, `recommend --explain`, `inspect-score` | contributions sum to score (test); explanations print for real users |
| **6 Novelty A+B** | 20–28 | adaptive-λ channel labels; faithfulness module | cold-start curve + faithfulness CSVs produced |
| **7 Experiments** | 20–28 | ablation, pruning, cold-start, channels, faithfulness, hand-judged sheet, figures | all CSVs + PNGs in `results/` and `figures/` |
| **8 Freeze + docs** | 28–32 | final numbers on test, README reproduction from a clean clone, docs complete | a teammate reproduces the demo from scratch using README only |
| **9 Video + report** | 28–36 | demo script, report notes, polish | submission checklist ticked (§17) |

Start the video and report by hour 28 at the latest.

**Suggested ownership (team of 4)** — freeze the data/interface contracts in hour 1:
- **A — Data & eval:** Phases 1, 3 (harness/metrics), 7.
- **B — Retrieval core:** Phase 2, TAAT/heap, champion lists, cluster pruning, parametric.
- **C — Hybrid & CF:** static quality, Jaccard, blend, tuning, Novelty A.
- **D — Explanations & faithfulness:** Phases 5–6B, hand-judged sheet, plus report/video coordination.

Each member must be able to explain their component in the video.

---

## 16. Report and video support

**Report (≤ 8 pages) — what each section needs from the code**

1. *Problem & track relevance:* cite Linden et al. 2003 (item-to-item CF), Hou et al. 2024 (Amazon Reviews 2023), Carbonell & Goldstein 1998 (MMR), Manning et al. *IIR*, optionally Zhou et al. 2018 (DIN). Tie to cold start / explanation / diversity.
2. *How we used IR:* pipeline diagram + `docs/ir_mapping.md` table.
3. *Beyond IR:* Bayesian rating average, hybrid blending, optional embeddings/MinHash; how each supports the IR side.
4. *Novelty:* Novelty A and B vs baseline centroid and "customers also bought".
5. *Evaluation:* main table, ablation, cold-start curve, pruning trade-off, faithfulness table, hand-judged P@5.
6. *Limitations & next steps:* §10.4 + roadmap (MMR, dense retrieval, attention, continue as course project).
7. *Work division* + *AI-use declaration* (from `docs/`).

**Video (5–8 min, no slides, nothing hard-coded)**
1. Problem + why T3 (≤ 1 min).
2. Live end-to-end: `search` and `recommend --explain` on real users, including **one limitation** (e.g. a cold user or a noisy-text item).
3. Internals: `inspect-term` (postings), `inspect-item` (weights), `inspect-score` (cosine, CF, λ, net).
4. Evaluation: main table, cold-start curve, one ablation, faithfulness.
5. Each member explains their component.

---

## 17. Definition of done / submission checklist

- [ ] `make all` (or the README's commands) reproduces data → index → tune → eval → experiments from a clean clone
- [ ] All tests pass (`make test`)
- [ ] Main table, ablation, cold-start curve, pruning curve, faithfulness table, hand-judged P@5 exist in `results/` + `figures/`
- [ ] README states what works and what is planned; dataset credited
- [ ] `docs/ir_mapping.md`, `docs/ai_use_log.md`, `docs/work_division.md` complete
- [ ] Demo shows live commands and intermediate output (postings, weights, scores) + one limitation
- [ ] Report PDF ≤ 8 pages with the seven sections, diagram, work division, AI-use declaration
- [ ] Repo link, video link (unlisted), report PDF submitted via the team form before the deadline
