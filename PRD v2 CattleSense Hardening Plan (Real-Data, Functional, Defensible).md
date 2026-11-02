# PRD v2: CattleSense Hardening Plan (Real-Data, Functional, Defensible)

Nov 3, 2026 · @Amodh

## 1. Diagnosis: why it reads as AI-generated

The project looks generated for five main reasons (the table lists eight symptoms): its headline results come from data you invented, the dashboard shows typed-in numbers, the models answer no single question, the repo cannot be rerun end to end, and the best real results were never stress-tested. All five are fixable in about six weeks. This PRD is based on your architecture write-up, not the code itself; rows marked *verify* need a check in the code.

| # | Symptom | Evidence | Why a reviewer distrusts it | Fix |
| --- | --- | --- | --- | --- |
| 1 | Circular results | Estrus, conception and lactation are trained and scored on generate\_data.py output (300 cows, rules you wrote) | A model that recovers rules you coded proves nothing about cows | Report only is\_synthetic=0 results; synthetic data is for pipeline tests (FR-A3, FR-B1) |
| 2 | Hardcoded dashboard | KPI header and cards typed into index.html; data.json built by scratch scripts such as run\_option5.py | Retrain, and the screen no longer matches the models | src/export\_web.py; every number read from reports/metrics.json (FR-D1, FR-D2) |
| 3 | Fake sandbox | Sandbox tab runs a JavaScript formula, not /v1/\* | The demo is not the system | Sandbox posts to the real API (FR-D3) |
| 4 | No single product | Behaviour classifier, jerk windows, estrus-duration regressor, ear-tag classifier, conception and lactation sit side by side | Reads as a pile of notebooks, not a system | One product, one question (section 2) |
| 5 | Untrusted constants | Numbers typed in from SupplementaryFigures.pdf stored beside trained parameters | Hand-entered values look like learned ones | Move to data/priors.yaml with citation; never used in metrics (FR-A5) |
| 6 | Unused or stale assets | Wagyu&Angus images unused; gen\_synthetic.py and 10-cow CSVs superseded; tests assert fixed row counts (2,663,644) | Tests that check the data's size do not check the model's behaviour | Use or delete; behaviour tests (FR-E3) |
| 7 | Unaudited high scores | Ear-tag ROC-AUC 0.954 (event vs diestrus); data2 macro-F1 0.979 on 3 test cows | Very high scores on small or easy tasks usually mean leakage or easy negatives. *Verify* | Leakage audit and harder evaluation (FR-A1, section 5) |
| 8 | Not reproducible | Not a git repo; web/data.json outside the pipeline; no CI | Nobody can rerun it and get your numbers | Git, one command rebuild, CI (FR-E1, FR-E2) |

## 2. Product definition and scope

CattleSense v2 is one product: an estrus alert that reads each animal's daily activity and temperature from an ear tag or collar and flags who is likely in heat today, with a measured false-alert rate. Estrus is the only task where you hold real labels (data4 PDF dates for 333 animals, Smid et al. events), so it is the only task that can carry an accuracy claim.

**Goals**

- G1: Estrus alert evaluated on animals never seen in training, reported as recall and false alerts per 100 animal-days with animal-level bootstrap intervals.
- G2: Beat the simple rule (activity above own baseline) by a margin agreed before the run, or report honestly that it does not.
- G3: Every number on screen traces to a file in reports/.
- G4: A clean clone rebuilds all results with one command in under 30 minutes.
- G5: Every dataset has a provenance record (source, licence, labels, allowed use).

**Non-goals:** a new deep-learning model, video or YOLO tracking, a mobile app, any accuracy claim for conception or lactation without real records.

**Scope tiers**

| Tier | Component | Data behind it | Claim you may make | Action |
| --- | --- | --- | --- | --- |
| A: core | Estrus alert (ear-tag activity and temperature) | data4 ear-tag plus PDF labels | Held-out animal recall and false-alert rate | Rebuild evaluation (section 5) |
| A: support | Behaviour features (eating, ruminating, walking) | data2 collar and halter, db-cow-walking | Held-out cow accuracy of behaviour labels; used as estrus features only if they improve A | Keep, but only as a feature source |
| A: insight | Estrus duration, indoor vs outdoor, sound vs lame | Smid et al., 94 events | Descriptive statistics from real events | Keep as a findings page; check target is not leaked (FR-A4) |
| B: simulation | Conception, lactation | generate\_data.py | None. Labelled SIMULATED everywhere | Demote to a Simulation Lab page; lactation can ship as a Wood-curve calculator the user feeds with real early milk |
| C: park | data3 jerk windows, Wagyu&Angus images, gen\_synthetic.py | n/a | None | Delete from the main path or move to /experiments with a README |

If you can obtain real insemination or milk records (a college dairy farm, a partner farm, a public dataset whose licence you have checked), tier B components move up to tier A and use the same protocol as section 5. Until then, saying so in the viva is stronger than a 0.9 score on invented cows.

## 3. Data requirements

Every dataset gets one row in data/registry.yaml, and the code refuses to train on anything not listed there. The registry is what lets you answer "where did this number come from" in one sentence.

| Dataset | What it is | Unit and size (from your write-up) | Labels | Allowed use | Split by |
| --- | --- | --- | --- | --- | --- |
| data4 ear-tag | Step counts and temperature, 3 sheets | 2,663,644 rows, 409 animals, 7,825 animal-days | Estrus date and AM/PM from PDF, 333 animals matched | Train and test the estrus alert | Animal |
| Smid et al. | Estrus behaviour events | 94 events, 45 cows | Estrus duration (SAS-valid) | Descriptive findings; duration model only if target is clean | Cow |
| data2 | 10 Hz collar and halter | 18 cows, about 7.9 GB | Eating, ruminating, other | Behaviour features; benchmark | Cow (official test cows 04, 10, 11) |
| db-cow-walking | 10 Hz labelled collar clips | Four classes | Walking, grazing, standing, resting | Behaviour benchmark | Cow or clip source |
| data3 | 1 Hz jerk stream | 12,654 rows | None used | Parked | n/a |
| generate\_data.py | Simulated herd | 300 cows | Simulated | Unit tests, demos, never reported accuracy | n/a |

**Requirements**

- FR-A0: data/registry.yaml lists source, URL or citation, licence, collection date, animals, labels, is\_synthetic and allowed use for each dataset. *Verify* licences for data2, data3 and data4 before publishing anything derived from them.
- Raw folders (data2, data3, data4) are read-only and kept out of git. A scripts/fetch\_data.py downloads or checks them against stored SHA-256 checksums.
- Every derived table carries is\_synthetic and a source column. Real and simulated rows never share a file.
- reports/data\_inventory.json also prints, per dataset, the number of positive labels and the number of animals with at least one positive. These two counts set how narrow your confidence intervals can be; write them next to every metric.
- Known gap to resolve: confirm species, breed and housing for the data4 animals, and how sensors were worn. The model card must state them, because a cow-herd result does not transfer to Wagyu or Angus beef cattle without evidence.

## 4. Functional requirements

Twenty-three requirements in four groups (A, B, D, E); each has a pass condition you can demonstrate. Group A and B change what you can claim, so do them first.

| ID | Requirement | Done when |
| --- | --- | --- |
| **A: Data audit** |  |  |
| FR-A0 | Data registry (section 3) | Training aborts on an unregistered dataset |
| FR-A1 | Leakage audit of the existing ear-tag result (AUC 0.954): check the baseline window excludes the label day, no animal in train and test, negatives are not only quiet mid-cycle days | A written audit in reports/audit\_data4.md with the corrected number, even if it is lower |
| FR-A2 | Clean-run reproduction of every number in your current README | Numbers match, or the README is corrected |
| FR-A3 | Synthetic rows excluded from every reported metric | A test fails if any is\_synthetic=1 row reaches evaluate.py |
| FR-A4 | Audit the Smid duration regressor (R² 0.839): confirm no feature encodes duration, e.g. end time minus start time | Features listed in the model card; R² rerun without suspect features |
| FR-A5 | Literature numbers from SupplementaryFigures.pdf moved to data/priors.yaml with page and figure citation | No hand-typed constant remains in src/ |
| **B: Modelling** |  |  |
| FR-B1 | All reported results use real data only | Metrics file has a data\_source field; dashboard shows it |
| FR-B2 | Estrus is evaluated as per animal-day detection over all days of each held-out animal, not event vs diestrus samples | Output is recall at a fixed false-alert rate |
| FR-B3 | Three models compared: activity-ratio rule, logistic regression, gradient boosting | One table, same folds, same threshold procedure |
| FR-B4 | Features use only data up to the day being scored | Leakage test: change a later value, the earlier features do not move |
| FR-B5 | Threshold fixed on validation animals before the test animals are touched | Threshold stored in the model file with its validation date |
| FR-B6 | Ablations: activity only, temperature only, both | A table showing whether temperature adds anything |
| FR-B7 | Model card per model: data, species, sample counts, metrics with intervals, known failure modes | reports/model\_cards.md generated, not typed |
| **D: Serving and dashboard** |  |  |
| FR-D1 | src/export\_web.py writes web/data.json from reports/ and out/ | Called at the end of train.py; make web exists |
| FR-D2 | No number typed in index.html | Search finds no metric literal; every KPI has a data-source attribute |
| FR-D3 | Sandbox posts rows to /v1/estrus and shows the returned score, threshold and model\_version | Network tab shows the call |
| FR-D4 | API validates the input schema and returns 422 with a reason for bad rows | Contract test covers missing fields, out-of-range values, empty batch |
| FR-D5 | Simulated outputs carry a visible SIMULATED banner and is\_synthetic in the JSON | Banner present on every tier B view |
| **E: Engineering** |  |  |
| FR-E1 | Git repository, .gitignore for raw data and models, README with exact commands | Fresh clone reaches a running dashboard by following the README |
| FR-E2 | make all runs fetch, validate, features, train, evaluate, export, test; CI runs a small fixture on every push | Green check on the repository |
| FR-E3 | Behaviour tests: leakage, label-shuffle (score falls to chance), determinism with fixed seeds, API contract | Tests fail when you deliberately break the property |
| FR-E4 | Pinned requirements.txt and a Python version | Install works on a clean machine |
| FR-E5 | Remove or archive dead code (gen\_synthetic.py, scratch scripts, unused models) | Nothing in src/ is unreferenced |

## 5. Evaluation protocol on real data

The test is whether the model flags heat in animals it has never seen, at a false-alert rate a farmer would tolerate. Write the pass rule down before running anything, so the result cannot be bent to fit.

**Task definition.** One row per animal-day for every day an animal was monitored. Positive = the day of a labelled estrus (the PDF gives date and AM/PM). An alert counts as a hit if it falls within one day of the labelled date. Everything else is a day with no estrus, including days close to estrus.

**Splits.** Group by animal, never by row or window.

1. Hold out 20% of animals as a final test set; touch it once.
2. On the remaining 80%, run 5-fold GroupKFold, repeated 3 times with different seeds.
3. Choose the alert threshold inside the training folds only (FR-B5).

**Metrics, always with animal-level bootstrap intervals (1,000 resamples)**

| Metric | Why |
| --- | --- |
| Event recall: share of labelled estrus events with an alert within one day | The farmer's question: did it catch the heat |
| False alerts per 100 animal-days | The farmer's cost: wasted checks |
| Precision-recall AUC with prevalence printed beside it | ROC-AUC looks high when positives are rare |
| Median lead time in hours before the labelled time | Late alerts miss the insemination window |
| Counts: positives, animals with a positive, total animal-days | Tells the reader how wide the interval must be |

**Leakage checklist (run before trusting any score)**

| Check | How | Pass |
| --- | --- | --- |
| Animal overlap | assert train and test animal sets are disjoint | Zero shared animals |
| Baseline window | rolling baseline ends the day before; shift test | Changing today's value does not change today's baseline |
| Future values | perturb values after day t and recompute features for day t | Features for day t identical |
| Easy negatives | compare AUC on all days vs only days within 3 days of an event | Drop is reported, not hidden |
| Label timing | confirm the labelled date and the sensor timestamps share a clock and time zone | Offsets documented |
| Duplicate rows | hash animal-day rows across sheets | No duplicate animal-days after merging the three sheets |

**Negative controls.** Shuffle labels within each animal: score must fall to chance. Shift labels by 7 days: score must fall sharply. If either holds up, the pipeline is leaking and the headline number is void.

**Pre-registered success rule (edit the numbers now, not after).** The gradient-boosting model counts as better than the activity-ratio rule only if, at the same false-alert rate, its event recall is higher by at least 10 points and the bootstrap interval of the difference excludes zero. If it is not, the report says the simple rule is as good, which is still a valid and honest result.

## 6. Target architecture and repo cleanup

Keep your pipeline layers, but make one path run from raw data to the screen, with nothing produced by hand. Data flows one way: raw, to registry-checked tables, to features, to models and reports, to the API and dashboard.

```text
Majorprj/
  README.md  Makefile  config.yaml  requirements.txt  .gitignore
  data/registry.yaml  data/priors.yaml     # provenance and cited constants
  scripts/fetch_data.py                    # checksums for raw folders
  src/
    ingest/   data2.py  data4.py  smid.py  # one reader per dataset, raw to tidy tables
    features.py   evaluate.py   audit.py   # audit.py runs the leakage checklist
    models/   estrus.py  behaviour.py
    train.py  predict.py  export_web.py    # export_web is new
    service.py                             # /healthz /v1/estrus, schema validated
  web/  server.py  index.html              # reads data.json only
  simulation/  generate_data.py  lactation.py  conception.py   # tier B, labelled
  experiments/  data3.py  gen_synthetic.py  images_note.md     # tier C, not in make all
  tests/  test_leakage.py  test_controls.py  test_api.py  test_determinism.py
  reports/  metrics.json  model_cards.md  audit_data4.md  data_inventory.json
```

**Changes from today**

| Change | Reason |
| --- | --- |
| Split data2\_data3.py and data4\_integration.py into one reader per dataset | Each file does ingest, features, modelling and plotting; none can be tested alone |
| Move simulated herd, conception and lactation under simulation/ | Keeps invented data physically separate from real results (FR-A3) |
| Add src/audit.py and tests/test\_controls.py | Makes the leakage checklist and shuffle control run on every build |
| Add src/export\_web.py and make web | Ends the hand-built data.json |
| Replace stdlib XLSX and PDF parsing only if fragile | Keep it if tests pass on the real files; a library is not required for authenticity |
| Make Wagyu&Angus either an experiment or removed | An unused folder in a submitted project invites questions |

**Serving contract.** POST /v1/estrus takes rows of cow\_id, date, activity, temperature and returns per row: score, alert, threshold, model\_version, data\_source. GET /healthz returns the model version and the training data hash, so the dashboard can show exactly which model produced each number.

## 7. Milestones

Six weeks, in this order because each gate protects the next: you cannot trust modelling results before the audit, and you cannot trust the dashboard before the models.

| Milestone | Week | Work | Requirements | Exit gate |
| --- | --- | --- | --- | --- |
| M0 Foundation | 1 | Git repo, registry, fetch script with checksums, pinned requirements, clean-run of current numbers | A0, A2, E1, E4 | Every number in the README is reproduced or corrected |
| M1 Audit | 1-2 | Leakage checklist on data4, Smid target check, priors file, shuffle and shift controls | A1, A4, A5, E3 | audit\_data4.md written; shuffle control falls to chance |
| M2 Honest evaluation | 2-3 | Per animal-day task, three models, ablations, animal bootstrap, apply the pre-registered rule | B1-B6 | Results table with counts and intervals, win or no win stated |
| M3 Restructure | 3-4 | Split dataset modules, move simulation, delete dead code, banners on simulated views | A3, D5, E5 | make all green on a fresh clone |
| M4 Live dashboard | 4-5 | export\_web.py, KPIs from metrics.json, sandbox calls the API, schema validation | D1-D4 | Retrain, then the dashboard changes with no file edited by hand |
| M5 Release | 5-6 | CI, generated model cards, README, demo script, viva notes | B7, E2 | CI green; a classmate reproduces your results from the README alone |

If time runs short, cut in this order: M5 polish, then M4 sandbox, then ablations. Never cut M1 or M2; they are what make the project real.

## 8. Acceptance criteria

The project is done when every row passes. Notice there is no target accuracy: with real data you accept whatever the protocol gives, and the criteria check that the number is honest, not that it is large.

| Criterion | Pass condition | Evidence file |
| --- | --- | --- |
| Real-only reporting | Every metric has data\_source = real; no is\_synthetic=1 row reaches evaluation | metrics.json, test\_leakage.py |
| Leakage tests | All checklist items automated and passing | CI log |
| Negative controls | Within-animal label shuffle gives PR-AUC inside the bootstrap interval of prevalence | test\_controls.py |
| Estrus result reported fully | Event recall, false alerts per 100 animal-days, PR-AUC with prevalence, lead time, all with intervals and counts | metrics.json |
| Model comparison decided | Pre-registered rule applied; the report states win or no win | reports/model\_cards.md |
| Dashboard integrity | Zero typed metrics in index.html; each KPI shows its source file and model\_version | export\_web.py output |
| Sandbox | Browser network call to /v1/estrus returns a model score | test\_api.py |
| Simulated content | SIMULATED banner on every tier B view | Visual check, screenshot in README |
| Rebuild | make all on a fresh clone finishes in under 30 minutes with raw data present | CI and README |
| Reproducibility | A second person reproduces the headline number within 0.01 using fixed seeds | Their run log |
| Documentation | One model card per shipped model with species, counts, limits | model\_cards.md |

## 9. Risks, open questions and the viva

The biggest risk is a good one: the audit may cut your ear-tag score substantially. That is the project working, not failing, and it is the most credible thing you can present.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Audit lowers the 0.954 AUC | Headline drops | Report the corrected number and the audit that produced it |
| Few estrus events per animal | Wide intervals | Repeat CV, print counts, claim no superiority the interval cannot support |
| Labels are AM/PM only | Hit definition is fuzzy | Report recall at tolerance 0 and 1 day |
| Dairy sensors, beef project | Result may not transfer to Wagyu or Angus | State species in the model card; make no transfer claim |
| Temperature confounded by weather or tag position | Signal looks useful but is not | Per-animal baseline plus the temperature ablation (FR-B6) |
| Raw data licences unclear | Cannot publish derived results | Check before M3; keep raw data out of the repository |
| Time overrun | Incomplete project | Cut order in section 7 |

**Open questions for you**

- Are the data4 animals dairy or beef, and how was estrus labelled (visual observation, mount detector, hormone)?
- What are the licences for data2, data3 and data4?
- Can you get any real insemination or milk records from a farm or college dairy unit? Even 50 cows with outcomes moves conception into tier A.
- Does your examiner expect a deployed product or a validated study? This PRD assumes the second.

**How to defend it in the viva**

- Open with the limit: "I had real estrus labels but no real conception or milk records, so those two are simulation tools and are labelled."
- Show the audit before the score: what you checked for leakage and what changed.
- Show the simple rule beside your model. If the model wins, show the interval; if it does not, say so.
- Run the label-shuffle control live. A score that collapses to chance is the clearest proof the pipeline is honest.
- Change one training setting, run make all, and let the dashboard update on its own.
