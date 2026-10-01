# M&A Acquirer Identification Engine

An explainable prototype for identifying ten candidate acquirers for a
private, regional Healthcare Services company with approximately $200M
enterprise value and strong EBITDA margins.

Python derives rankings and evidence from the supplied 500-transaction
CSV. Gemini synthesizes buyer-specific narratives. Streamlit presents
the rankings, six-section memos, supporting transactions and downloads.

The output supports an initial buyer-screening discussion. It does not
establish current buyer interest or predict acquisition probability.

## Quick start

Requires Python 3.11+.
From the project root, install dependencies and launch:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

An isolated environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

The supplied CSV should be at:

```text
data/ma_transactions_500.csv
```

### Gemini configuration

Rankings and evidence do not require an API key. Matching saved memos
can also be inspected without making provider calls.

For fresh generation, copy .env.sample to .env in the project root and replace your_gemini_api_key_here with your Gemini API key.

```text
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Generate missing or stale memos:

```powershell
python src/rationale.py
```

Alternatively, click **Refresh All Ten Memos** in Streamlit. This
generates all ten again, including buyers with existing cached memos.

The prototype was tested using the Gemini free tier. Fresh generation
requires an API key and is subject to provider quotas and availability.
Do not commit `.env` or API keys.

## Banker workflow and outputs

1. Review the target assumptions and ranked buyer list.
2. Compare historical fit, comparable-deal counts and conviction.
3. Select a buyer to inspect its memo.
4. Inspect completed precedents, buyer comps, market comps and
   historical process observations.
5. Download an individual memo or the current memo collection.

Each memo covers:

| Section | Content |
|---|---|
| Acquirer overview | Buyer type, observed transaction scale and dated activity |
| Strategic fit thesis | Target-specific interpretation of historical evidence |
| Precedent activity | Targets, transaction IDs, sectors, EVs, deal types and multiples |
| Valuation context | Closed-deal EV/EBITDA and EV/Revenue comparisons |
| Risk flags | At least two supported constraints or diligence items |
| Conviction | High, Medium or Low with a data-based explanation |

A banker angle summarizes the outreach implication.

Memos are exported as structured JSON and Markdown:

```text
outputs/acquirer_rationales.json
outputs/acquirer_rationales.md
```

These sample outputs are generated narratives, not manually authored
buyer-specific replacements.

## Architecture and responsibilities

The pipeline is:

```text
CSV → transaction fit → buyer ranking → evidence packages
    → Gemini structured response → validation → memos and UI
```

| Module | Responsibility |
|---|---|
| `src/config.py` | Supplied target profile |
| `src/data_loader.py` | CSV loading, required fields and data validation |
| `src/scoring.py` | Sector, size and margin fit; buyer aggregation |
| `src/evidence.py` | Scoped precedents, valuation summaries and process evidence |
| `src/rationale.py` | Prompt, JSON schema, checks, retry, caching and rendering |
| `src/validation.py` | Historical buyer-recovery evaluation and baselines |
| `src/app.py` | Interactive rankings, memos, evidence and downloads |

Python performs arithmetic and evidence selection. Gemini synthesizes
the buyer overview, banker angle, fit thesis, precedent description
and valuation interpretation.

Conviction and eligible risk wording are controlled by code. This
division makes numerical and evidence rules reproducible while retaining
LLM synthesis for the qualitative memo.

## Ranking methodology

### Transaction fit

Three signals form the transaction fit score:

```text
fit = 0.50 × sector_similarity
    + 0.25 × size_similarity
    + 0.25 × margin_similarity
```

**Sector similarity**

The exact target sector receives 1. Cross-sector similarity uses
Jaccard overlap between the sets of observed buyers in each sector:

```text
sector_similarity = shared_buyers / buyers_in_either_sector
```

This captures historical association between sectors. It is not proof
of clinical adjacency or strategic fit. Sponsor participation in
multiple sectors can create overlap without corresponding operating
similarity.

**Deal-size similarity**

```text
size_similarity = exp(-abs(log(historical_EV / target_EV)))
```

This treats reciprocal size differences symmetrically. Against a $200M
target, both a $100M deal and a $400M deal receive a size score of 0.5.
Missing or nonpositive historical EV contributes zero size fit.

**EBITDA-margin similarity**

```text
margin_similarity =
    max(0, 1 - abs(historical_margin - target_margin) / target_margin)
```

The target's actual margin is not supplied. “Strong” is represented
by the target-sector 75th percentile of historical EBITDA margins.
This is a modeling proxy, not a claim about the target's financials.

### Buyer aggregation and evidence adjustment

Only closed transactions directly contribute to buyer ranking.
Each buyer's base fit is the mean of its three strongest completed
precedents, or the available number if fewer than three exist.

Sparse histories are adjusted toward a common prior:

```text
depth = min(sum(sector_similarity across closed buyer deals), 6)

reliability = depth / (depth + 2)

buyer_score = reliability × base_fit
            + (1 - reliability) × prior
```

The prior is the mean base fit across buyers. Sector-weighted depth
gives more credit to relevant histories, and the cap limits the effect
of transaction volume.

The weights, adjustment strength of 2 and depth cap of 6 are transparent
heuristics. They are not learned or claimed to be optimal. Ties are
resolved alphabetically.

The buyer list is derived from the data. There is no hardcoded ranking
or required strategic/sponsor mix.

## Evidence selection and valuation

Ranking precedents and memo comparables serve different purposes.

- Ranking precedents are the highest-scoring completed transactions.
- Memo comparables prioritize completed transactions near the target
  EV, so the narrative does not rely solely on much larger or smaller
  transactions.

### Buyer comparable scope

For this target, size-matched buyer comps are closed deals at
$100–400M EV in:

- Healthcare Services
- Physician Groups
- Behavioral Health
- Home Health/Hospice
- Dental

This half-to-double EV band and provider-sector group are explicit
analytical assumptions, not a learned adjacency model.

Memo selection prefers non-minority transactions, then exact-sector
relevance and proximity to target EV. At least two selected precedents
are required when available; a single comparable is used when that
is the available evidence.

When there are no matching comps, broader completed precedents are
identified as outside the size band. Their existence does not create
size-matched evidence.

### Market comparable scope

Market comps are selected independently of the buyer:

1. Use closed exact-sector transactions in the same EV band.
2. If fewer than five are available, expand to the disclosed
   provider-sector group.

Python calculates medians, interquartile ranges and nonmissing
observation counts for EV/EBITDA and EV/Revenue.

Buyer medians use the defined size-matched provider-sector sample.
If that sample is empty, unrelated buyer medians are not substituted.
Buyer and market scopes can differ, which limits interpretation of
their valuation gaps.

Historical multiples are valuation context, not a target price,
a bidding forecast or proof of willingness to pay.

## Rank versus conviction

Rank measures relative historical fit under the scoring formula.
Conviction measures the depth of size-matched historical evidence.

| Conviction | Size-matched provider-sector closed deals |
|---|---:|
| High | 4 or more |
| Medium | 1–3 |
| Low | 0 |

The tier and count explanation are calculated in Python. Gemini
cannot change the tier.

A lower-ranked buyer can have High conviction because it has numerous
comparables but a weaker relative fit score. A higher-ranked buyer can
have Low conviction because its historical fit scores well while its
size-matched evidence is sparse.

High conviction does not establish current appetite, operational fit,
funding capacity or acquisition probability. Adjacent-sector experience
and minority investments retain their comparability limitations.

## LLM prompt, validation and caching

The prompt supplies:

- The target profile and margin-proxy caveat.
- Buyer history with clearly defined evidence scopes.
- Selected memo precedents and transaction details.
- Precomputed valuation comparisons.
- The assigned conviction and explanation.
- Eligible risk options and historical outcome counts.

Gemini returns structured JSON. Code renders consistent memo headings
and adds both valuation summaries.

Eligible risk options are generated from evidence such as sample depth,
adjacent-sector dependence, minority-investment comparability,
historical valuation gaps and unexplained withdrawn/terminated outcomes.

When non-closing outcomes exist, a process-history diligence item is
included. Its wording does not assign the cause to buyer failure.
Historical bidder counts remain evidence, not proof of current
competitive tension.

Standardized risk wording improves consistency and auditability.
It can also make some sections repetitive; buyer-specific synthesis
remains important in the narrative fields.

### Validation and failure handling

Targeted checks cover:

- Required nonempty fields and risk structure.
- The assigned conviction tier and explanation.
- Supplied closed-deal citations and buyer ownership of risk citations.
- Selected precedent coverage and outside-band labeling.
- Eligible, distinct risk categories.
- Selected financial-number and formatting constraints.
- Declarative fit theses and avoidance of verbatim risk-action repetition.

An invalid response receives one corrective retry containing the
previous response and validation findings. A second failure raises an
error rather than silently accepting invalid output.

These checks do not prove semantic accuracy. For example, a memo can
cite valid transactions but still conflate their sectors or infer
ownership implications unsupported by the data.

### Non-determinism and resuming

Generation uses a low temperature of 0.2, but output is not guaranteed
to be deterministic.

A cache fingerprint includes the target, evidence, model, prompt
version and derived inputs. Matching saved responses are validated
before reuse. Substantive prompt changes should increment
`PROMPT_VERSION`.

Each completed memo is saved, allowing interrupted CLI generation to
resume missing or stale buyers. A full app refresh intentionally
generates a new batch.

## Historical evaluation

Run:

```powershell
python src/validation.py
```

The evaluation holds out closed transactions from 2018 onward and uses
strictly earlier-year history. Every method uses the same eligible
tests and candidate universe.

There are 276 eligible tests and 53 exclusions. A test is excluded
when the actual buyer or target sector has no earlier history.
Candidates are buyers with earlier closed transactions. An actual
buyer with only earlier non-closed history remains an eligible test
but is a miss for each method.

Hit@K measures whether the observed buyer appears in the top K.

| Method | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|
| Random, analytical expectation | 1.4% | 7.2% | 14.3% |
| Prior closed-deal count | 5.8% | 23.6% | 45.3% |
| Sector-weighted closed-deal count | 7.6% | 21.7% | 44.6% |
| Original scoring | 5.1% | 24.3% | 45.3% |
| Log-size only | 4.3% | 25.0% | 45.3% |
| Shrinkage only | 7.2% | 26.8% | 44.9% |
| Log-size + shrinkage: current | 6.2% | 26.1% | 44.2% |

The current model improves Hit@1 and Hit@5 relative to original scoring,
but reduces Hit@10 by 1.1 percentage points. It does not outperform the
popularity baseline overall.

The current approach is retained for symmetric size comparisons and
more cautious treatment of sparse histories, not because the evaluation
establishes superior predictive performance. The results are descriptive;
statistical significance and optimal parameters are not established.

The held-out deal's realized EV and margin are used as its target
description. This is a historical association check, not a simulation
of information available before announcement. Repeated buyers,
correlated deals, exclusions and semi-fictional targets limit
generalization.

This evaluation tests buyer recovery, not narrative accuracy or current
buyer interest.

## Runtime and observed checks

A full refresh of all ten memos took approximately 40 seconds in one
local, user-timed run using the Gemini free tier. This supports the
under-60-second goal for that run, not a latency guarantee.

The app was observed loading saved memos, displaying the ranking and
conviction definitions, switching buyers and showing comparable-deal
tables. Generated output was reviewed against the supplied CSV.

Fresh generation remains dependent on provider availability, quotas,
network latency and corrective retries.

## Assumptions and known limitations

- The supplied CSV is the primary source of truth. No external
  corporate database or current company research is used.
- Acquirer names are treated as consistent buyer entities. Target
  names are semi-fictional and historical coverage is incomplete.
- Sponsor-heavy coverage influences sector overlap and buyer activity.
  Cross-sector co-occurrence is not necessarily operating adjacency.
- The target's actual EBITDA margin, detailed business model and
  specific region are unavailable.
- Private, regional and mid-market descriptions do not establish
  geographic overlap, ownership compatibility or integration fit.
- Deal types are not weighted in the ranking formula. Minority,
  SPAC, recapitalization and control transactions can have different
  relevance despite comparable EV.
- Historical financing types describe consideration; they do not
  establish present funding capacity.
- Withdrawn and terminated outcomes have unknown causes. Days to
  close do not establish regulatory or execution problems.
- Numerical and citation checks are partial. Narrative sector scope,
  causal statements and ownership implications can still be wrong.
  Generated memos require analyst review.
- The prototype exports structured text and Markdown; it does not
  enforce print pagination into a fixed one-page PDF.
- Buyer names are treated as separate entities as supplied; current
  corporate relationships and operating status are not screened.
  Financing labels are used as recorded and require verification
  against the buyer's legal structure before informing transaction advice.

## Improvements with more time

1. **Claim-level grounding:** link sector, transaction-type and ownership
   assertions to specific cited rows. Validate those relationships,
   retry only affected sections and flag unresolved claims for review.
2. **Narrative evaluation:** build a small evaluation set for factual
   attribution, buyer differentiation, actionability and unsupported
   inference, rather than relying on citation validity alone.
3. **Ranking sensitivity:** evaluate sector adjacency assumptions,
   transaction-structure comparability and shrinkage parameters on
   additional data.
4. **Simpler generation code:** consolidate overlapping prompt and
   validation rules while preserving checks supported by observed
   failure cases.
5. **Operational robustness:** improve quota/error handling and measure
   latency across multiple runs before optimizing concurrency.
6. **Buyer eligibility checks:** resolve related buyer entities and
   verify current operating status before presenting an actionable
   shortlist. Check financing labels against legal structure and
   flag questionable records for analyst review.
