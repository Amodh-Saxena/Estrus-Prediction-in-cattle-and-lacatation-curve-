# CattleSense v2: Updated Directory and Step-by-Step Implementation

Nov 3, 2026 · @Amodh

## 1. Updated directory

The new layout separates real-data code from simulation, gives every dataset its own reader, and puts everything the dashboard shows behind one export step. Tags: **NEW**, **MOVE** (exists today, new location), **SPLIT** (one file becomes several), **KEEP**, **ARCHIVE** (out of the main path).

```text
Majorprj/
  README.md                      KEEP    exact run commands, results copied from reports/
  Makefile                       KEEP    rewritten, see step 12
  config.yaml                    KEEP    seeds, split sizes, alert tolerance, thresholds
  requirements.txt               KEEP    pin versions (pip freeze)
  .gitignore                     NEW     data2/ data3/ data4/ models/ out/ *.db
  .github/workflows/ci.yml       NEW     runs make test on a small fixture
  PRD v2.md                      NEW     the PRD from the first doc

  data/
    registry.yaml                NEW     provenance per dataset
    priors.yaml                  NEW     cited constants from the supplementary PDF
    interim/                     NEW     tidy tables written by ingest (gitignored)
    fixtures/                    NEW     tiny hand-made CSVs for tests (committed)

  scripts/
    fetch_data.py                NEW     verify raw folders by SHA-256
    repro_current.py             NEW     step 2, reruns your old numbers

  src/
    __init__.py
    common.py                    NEW     config, paths, registry loader, seeding
    ingest/
      __init__.py
      data4_eartag.py            SPLIT   from data4_integration.py (3-sheet stream to animal-day)
      data4_labels.py            SPLIT   PDF estrus lookup
      smid.py                    SPLIT   Smid workbook reader
      data2_collar.py            SPLIT   from data2_data3.py
      cow_walking.py             MOVE    clip loader from models/behaviour.py
    features.py                  KEEP    estrus features only; causal, tested
    evaluate.py                  KEEP    grouped splits, bootstrap, metrics
    audit.py                     NEW     leakage checklist, writes reports/audit_data4.md
    models/
      estrus.py                  KEEP    GBM, logistic, rule baseline
      behaviour.py               KEEP    behaviour classifier only
    train.py                     KEEP    orchestrator, real data only
    predict.py                   KEEP    batch scoring
    export_web.py                NEW     reports/ to web/data.json
    service.py                   KEEP    schema-validated /v1/estrus

  simulation/                    MOVE    everything invented, labelled SIMULATED
    generate_data.py             MOVE
    build_database.py            MOVE
    lactation.py                 MOVE    from src/models/
    conception.py                MOVE    from src/models/

  experiments/                   ARCHIVE not run by make all
    gen_synthetic.py
    data3_jerk.py
    breed_images_note.md

  web/
    server.py                    KEEP
    index.html                   KEEP    remove typed numbers, add SIMULATED banner
    data.json                    KEEP    generated, gitignored

  tests/
    conftest.py                  NEW     fixture loaders
    test_leakage.py              NEW
    test_controls.py             NEW
    test_features_causal.py      NEW
    test_api.py                  NEW
    test_determinism.py          NEW

  reports/                       generated: metrics.json, model_cards.md,
                                 audit_data4.md, data_inventory.json
  models/  out/                  generated, gitignored
  data2/ data3/ data4/ db-cow-walking-main/ Wagyu&Angus/   raw, read-only, gitignored
```

**Deleted outright:** the scratch scripts that built data.json (run\_option5.py and similar), and any root-level copy of a module that now lives under src/. Keep one copy of each file.

## 2. Steps 0-2: foundation

Do these before touching any model. They make every later number traceable.

**Step 0: put the project under git (30 minutes)**

```bash
cd D:/Projects/Majorprj
git init
printf "data2/\ndata3/\ndata4/\ndb-cow-walking-main/\nWagyu&Angus/\nmodels/\nout/\n*.db\nweb/data.json\ndata/interim/\n__pycache__/\n" > .gitignore
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt && pip freeze > requirements.txt
git add -A && git commit -m "Baseline before v2 hardening"
git tag v1-baseline
```

The v1-baseline tag matters: you can always show what changed and why the numbers moved.

**Step 1: data registry and a hard gate**

Create data/registry.yaml. Fill the fields marked verify from the dataset pages or the supervisor; do not guess.

```yaml
datasets:
  data4_eartag:
    source: "<citation or URL>"
    licence: "<verify>"
    species: "<verify: dairy or beef>"
    labels: "estrus date and AM/PM from PDF"
    is_synthetic: 0
    allowed_use: [estrus_train, estrus_test]
  smid_estrus:
    source: "<citation>"
    licence: "<verify>"
    is_synthetic: 0
    allowed_use: [findings]
  data2_collar:
    source: "Zenodo 4064802"
    licence: "<verify>"
    is_synthetic: 0
    allowed_use: [behaviour_train, behaviour_test]
  simulated_herd:
    source: "simulation/generate_data.py"
    is_synthetic: 1
    allowed_use: [unit_tests, demo]
```

Then src/common.py, which every stage imports:

```python
import random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text())

def require(dataset, use):
    """Abort unless the dataset is registered and approved for this use."""
    reg = yaml.safe_load((ROOT / "data/registry.yaml").read_text())["datasets"]
    if dataset not in reg:
        raise SystemExit(f"{dataset} is not in data/registry.yaml")
    if use not in reg[dataset]["allowed_use"]:
        raise SystemExit(f"{dataset} is not approved for {use}")
    return reg[dataset]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
```

scripts/fetch\_data.py stores a SHA-256 for each raw file in data/checksums.json (run once with --write) and later fails if a file changed. For the 7.9 GB data2 folder, hash in 1 MB blocks.

```python
import hashlib, json, sys
from pathlib import Path

RAW = ["data2", "data3", "data4", "db-cow-walking-main"]
OUT = Path("data/checksums.json")

def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()

files = {str(p): sha256(p) for d in RAW for p in Path(d).rglob("*") if p.is_file()}
if "--write" in sys.argv:
    OUT.write_text(json.dumps(files, indent=1))
else:
    saved = json.loads(OUT.read_text())
    bad = [k for k in saved if files.get(k) != saved[k]]
    sys.exit(f"changed or missing: {bad}" if bad else 0)
```

**Step 2: reproduce your current numbers from a clean run**

Before changing anything, find out which of your reported numbers are real. scripts/repro\_current.py runs the old pipeline and compares:

```python
import json, subprocess

CLAIMED = {  # copy each from your README or dashboard
    "eartag_roc_auc": 0.9544, "eartag_pr_auc": 0.8605,
    "data2_cv_macro_f1": 0.8793, "data2_test_macro_f1": 0.9786,
    "smid_r2": 0.8391,
}
subprocess.run(["python", "-m", "src.train", "--config", "config.yaml"], check=True)
metrics = json.load(open("reports/metrics.json"))
# adjust the lookup paths to your metrics.json layout
GOT = {k: metrics.get(k) for k in CLAIMED}
for k, v in CLAIMED.items():
    ok = GOT[k] is not None and abs(GOT[k] - v) < 0.005
    print(f"{k:22s} claimed {v:.4f} got {GOT[k]} {'OK' if ok else 'MISMATCH'}")
```

Exit gate: every line says OK, or you correct the README. Commit with the message "Reproduced v1 numbers".

## 3. Steps 3-4: ingest and audit

**Step 3: one reader per dataset, one shared animal-day table**

Split data4\_integration.py so each reader only turns raw files into a tidy table. Estrus modelling reads one table: data/interim/eartag\_animal\_day.csv, one row per animal per day.

| Column | Meaning |
| --- | --- |
| animal\_id, date | Key, unique together |
| steps\_sum, temp\_mean, temp\_q90, temp\_max, temp\_std, n\_readings | Daily sensor summaries from the three sheets |
| estrus | 1 on the labelled estrus date, otherwise 0 |
| session | AM or PM from the PDF, empty when estrus is 0 |
| source | Dataset id from the registry |

The label join lives in src/ingest/data4\_labels.py and keeps only animals that appear in the PDF, because an animal with no PDF entry has unknown labels, not negative ones.

```python
import pandas as pd
from src.common import require

def build_animal_day(daily, labels):
    require("data4_eartag", "estrus_train")
    daily = daily.assign(date=pd.to_datetime(daily["date"]).dt.normalize())
    lab = labels.assign(date=pd.to_datetime(labels["date"]).dt.normalize(), estrus=1)
    df = daily.merge(lab[["animal_id", "date", "estrus", "session"]],
                     on=["animal_id", "date"], how="left")
    df["estrus"] = df["estrus"].fillna(0).astype(int)
    df = df[df["animal_id"].isin(labels["animal_id"].unique())].copy()
    assert not df.duplicated(["animal_id", "date"]).any(), "duplicate animal-days"
    df["source"] = "data4_eartag"
    return df.sort_values(["animal_id", "date"]).reset_index(drop=True)
```

Check two things by hand and write the answers in the registry: whether the PDF lists every estrus during each animal's monitoring window (if it lists only some, days without a label are not truly negative), and whether the sensor clock and PDF dates share a time zone.

**Step 4: leakage audit of the ear-tag result**

src/audit.py takes out-of-fold predictions (columns animal\_id, date, estrus, p) from your current model and runs the checks from the PRD. Write the output to reports/audit\_data4.md.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def animal_overlap(train_ids, test_ids):
    shared = set(train_ids) & set(test_ids)
    assert not shared, f"{len(shared)} animals in both train and test"

def days_to_nearest_event(df):
    out = pd.Series(np.inf, index=df.index)
    for _, g in df.groupby("animal_id"):
        ev = g.loc[g["estrus"] == 1, "date"].to_numpy("datetime64[D]")
        if len(ev):
            d = g["date"].to_numpy("datetime64[D]")
            out[g.index] = np.abs(d[:, None] - ev[None, :]).min(axis=1).astype(int)
    return out

def hard_negative_report(oof, k=3):
    """AUC on all days vs only days within k days of an event."""
    dist = days_to_nearest_event(oof)
    near = oof[dist <= k]
    return {
        "auc_all_days": roc_auc_score(oof["estrus"], oof["p"]),
        "auc_near_event": roc_auc_score(near["estrus"], near["p"]),
        "pr_auc_all_days": average_precision_score(oof["estrus"], oof["p"]),
        "prevalence": oof["estrus"].mean(),
        "n_pos": int(oof["estrus"].sum()),
        "n_animals_pos": int(oof.loc[oof["estrus"] == 1, "animal_id"].nunique()),
    }

def label_shuffle_control(oof, seed=0, reps=200):
    """Shuffle labels inside each animal; the AUC should sit near 0.5."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(reps):
        y = oof.groupby("animal_id")["estrus"].transform(lambda s: rng.permutation(s.values))
        aucs.append(roc_auc_score(y, oof["p"]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 97.5))
```

How to read the result:

- **auc\_near\_event far below auc\_all\_days:** the model mostly separates quiet days from busy days, not heat from near-heat. Report the near-event number as the honest one.
- **Shuffled AUC well above 0.5:** the model is learning something about animals or dates that survives shuffling, such as an animal-level offset. Find it before going on.
- **Duplicate animal-days after merging the three sheets:** the assert in build\_animal\_day fires; fix the ingest.

Exit gate: reports/audit\_data4.md contains the corrected numbers and the counts, whatever they turn out to be.

## 4. Steps 5-6: features and evaluation harness

**Step 5: causal features (src/features.py)**

Every feature for day t may use data up to day t only, and the baseline must end on day t-1. Rolling windows work on rows, so first put each animal on a complete daily grid; otherwise a gap silently stretches the 14-day baseline.

```python
import numpy as np
import pandas as pd

SIGNALS = ["steps_sum", "temp_mean", "temp_q90"]
MIN_SCALE = {"steps_sum": 1.0, "temp_mean": 0.05, "temp_q90": 0.05}  # stops divide-by-near-zero

def to_daily_grid(df):
    parts = []
    for a, g in df.groupby("animal_id"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        parts.append(g.set_index("date").reindex(full).rename_axis("date")
                      .assign(animal_id=a).reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["estrus"] = out["estrus"].fillna(0).astype(int)
    return out

def _mad(w):
    return np.median(np.abs(w - np.median(w)))

def _days_since(flag):
    out, last = [], None
    for i, f in enumerate(flag.to_numpy()):
        out.append(np.nan if last is None else i - last)
        if f:
            last = i
    return pd.Series(out, index=flag.index)

def add_features(df, window=14, min_periods=5, alert_ratio=1.5):
    df = to_daily_grid(df).sort_values(["animal_id", "date"]).reset_index(drop=True)
    g = df.groupby("animal_id")
    for s in SIGNALS:
        past = g[s].transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).median())
        mad = g[s].transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods)
                                       .apply(_mad, raw=True))
        scale = np.maximum(1.4826 * mad, MIN_SCALE[s])
        df[f"{s}_ratio"] = df[s] / past
        df[f"{s}_z"] = (df[s] - past) / scale
        df[f"{s}_z_lag1"] = df.groupby("animal_id")[f"{s}_z"].shift(1)
    df["steps_z_max3"] = df.groupby("animal_id")["steps_sum_z"].transform(
        lambda x: x.rolling(3, min_periods=1).max())
    spike = (df["steps_sum_ratio"] > alert_ratio).fillna(False)
    df["days_since_spike"] = spike.groupby(df["animal_id"]).transform(_days_since)
    return df

FEATURES = [c for s in SIGNALS for c in (f"{s}_ratio", f"{s}_z", f"{s}_z_lag1")] \
           + ["steps_z_max3", "days_since_spike"]
```

Gradient boosting accepts missing values, so leave early days with too little history as NaN. Rows with a missing sensor reading stay in the table; do not fill them with zeros.

tests/test\_features\_causal.py proves causality: compute features, change every value after day t for one animal, recompute, and assert that all rows up to and including t are identical (use pandas.testing.assert\_frame\_equal on the FEATURES columns).

**Step 6: evaluation harness (src/evaluate.py)**

Four pieces: animal-level splits, an event-level metric, a threshold picker, and an animal bootstrap. Move days\_to\_nearest\_event from audit.py into this file and import it back.

```python
import numpy as np
import pandas as pd

def split_animals(ids, test_frac, seed):
    ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = max(1, int(round(len(ids) * test_frac)))
    return ids[n:], ids[:n]          # dev, test

def repeated_animal_folds(ids, k=5, repeats=3, seed=0):
    ids = np.array(sorted(set(ids)))
    for r in range(repeats):
        perm = np.random.default_rng(seed + r).permutation(ids)
        chunks = np.array_split(perm, k)
        for i in range(k):
            yield r, i, np.concatenate(chunks[:i] + chunks[i + 1:]), chunks[i]

def event_metrics(df, alert, tol=1):
    """df: animal_id, date, estrus; alert: boolean array aligned with df."""
    d = df.reset_index(drop=True).assign(alert=np.asarray(alert, bool))
    dist = days_to_nearest_event(d)
    events = d[d["estrus"] == 1]
    hits = 0
    for _, e in events.iterrows():
        w = d[(d["animal_id"] == e["animal_id"]) &
              ((d["date"] - e["date"]).abs() <= pd.Timedelta(days=tol))]
        hits += int(w["alert"].any())
    false_alerts = int(((d["alert"]) & (dist > tol)).sum())
    return {"event_recall": hits / max(1, len(events)), "n_events": int(len(events)),
            "false_alerts_per_100": 100 * false_alerts / max(1, len(d))}

def pick_threshold(df, p, max_fa=5.0, tol=1):
    """Highest recall with false alerts <= max_fa per 100 animal-days. Use validation scores only."""
    best_t, best_r = float(np.max(p)) + 1e-9, -1.0
    for t in np.unique(np.quantile(p[~np.isnan(p)], np.linspace(0.5, 0.999, 200))):
        m = event_metrics(df, p >= t, tol)
        if m["false_alerts_per_100"] <= max_fa and m["event_recall"] > best_r:
            best_t, best_r = float(t), m["event_recall"]
    return best_t

def bootstrap_by_animal(df, alert, tol=1, reps=1000, seed=0):
    d = df.reset_index(drop=True).assign(alert=np.asarray(alert, bool))
    groups = {a: g for a, g in d.groupby("animal_id")}
    ids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([groups[a].assign(animal_id=f"{a}_{i}") for i, a in enumerate(pick)],
                      ignore_index=True)
        out.append(event_metrics(s, s["alert"].to_numpy(), tol))
    r = pd.DataFrame(out)
    return {c: (float(r[c].quantile(.025)), float(r[c].quantile(.975)))
            for c in ("event_recall", "false_alerts_per_100")}
```

The bootstrap renames each resampled animal so a repeated animal counts as a separate animal in the event matching. For speed, run 200 reps while developing and 1,000 for the final report.

With only daily rows, lead time is measured in days from the labelled date (0 = same day, -1 = a day early), not hours. State that in the model card.

Exit gate: running event\_metrics on a perfect predictor gives recall 1.0 and 0 false alerts, and on an all-alert predictor gives recall 1.0 with a large false-alert rate. Put both as unit tests.

## 5. Steps 7-8: models, ablations, controls

**Step 7: three models on the same folds (src/models/estrus.py)**

The rule baseline and the two learned models share one interface, so the comparison is fair. The threshold is picked once, from out-of-fold scores on the development animals; the held-out test animals are scored only at the end.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluate import repeated_animal_folds
from src.features import FEATURES

FEATURE_SETS = {
    "activity":    [c for c in FEATURES if c.startswith("steps") or c == "days_since_spike"],
    "temperature": [c for c in FEATURES if c.startswith("temp")],
    "both":        FEATURES,
}

def make_model(name, seed):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "gbm":   # shallow and regularised: there are few positives
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              l2_regularization=1.0, min_samples_leaf=40,
                                              random_state=seed)
    raise ValueError(name)

def score(name, model, X, feats):
    if name == "rule":                      # activity above own baseline
        return X["steps_sum_ratio"].fillna(1.0).to_numpy()
    return model.predict_proba(X[feats])[:, 1]

def oof_scores(df, feats, name, cfg):
    acc, cnt = np.zeros(len(df)), np.zeros(len(df))
    for _, _, tr, te in repeated_animal_folds(df["animal_id"].unique(),
                                              cfg["cv_folds"], cfg["cv_repeats"], cfg["seed"]):
        mtr, mte = df["animal_id"].isin(tr).to_numpy(), df["animal_id"].isin(te).to_numpy()
        model = None if name == "rule" else make_model(name, cfg["seed"]).fit(
            df.loc[mtr, feats], df.loc[mtr, "estrus"])
        acc[mte] += score(name, model, df.loc[mte], feats)
        cnt[mte] += 1
    return acc / np.maximum(cnt, 1)
```

The driver loops over name in rule, logistic, gbm and feature set in activity, temperature, both (the rule ignores the set). For each pair it does this, in order:

1. Compute out-of-fold scores on the development animals.
2. pick\_threshold on those scores with max\_fa from config.yaml (start at 5 false alerts per 100 animal-days).
3. Fit on all development animals, score the test animals, apply the stored threshold.
4. Record event\_metrics and bootstrap\_by\_animal for the test animals, plus PR-AUC and prevalence.

The PRD's decision rule then needs one more function: a paired animal bootstrap of the recall difference between gbm and the rule at their own thresholds. Resample animals once per replicate, apply both alert vectors to the same resample, and store the difference. Report that the model wins only if the lower 2.5% bound is at least +0.10, otherwise write "no demonstrated gain over the rule".

Exit gate: a table with 7 rows (rule, 3 logistic, 3 gbm), each with recall, false alerts per 100 animal-days, intervals, n\_events, n\_animals.

**Step 8: negative-control tests (tests/ and conftest.py)**

The fixture is a tiny planted-signal herd, used only for tests, so you can prove your checks can fail.

```python
# tests/conftest.py
import numpy as np, pandas as pd, pytest

def make_fixture(n=40, days=60, seed=0):
    rng, rows = np.random.default_rng(seed), []
    for a in range(n):
        steps = rng.normal(100, 15, days).clip(10)
        temp = rng.normal(38.6, 0.15, days)
        estrus = np.zeros(days, int)
        for d in range(10, days - 5, 21):
            e = d + int(rng.integers(0, 3))
            estrus[e] = 1; steps[e] *= 2.0; temp[e] += 0.3
        rows.append(pd.DataFrame({"animal_id": f"a{a}",
            "date": pd.date_range("2025-01-01", periods=days),
            "steps_sum": steps, "temp_mean": temp, "temp_q90": temp + 0.2,
            "estrus": estrus, "source": "fixture"}))
    return pd.concat(rows, ignore_index=True)

@pytest.fixture
def herd():
    return make_fixture()
```

```python
# tests/test_controls.py
import numpy as np
from sklearn.metrics import roc_auc_score
from src.features import add_features, FEATURES
from src.models.estrus import oof_scores

CFG = {"cv_folds": 4, "cv_repeats": 1, "seed": 0}

def test_signal_is_found_and_shuffle_kills_it(herd):
    df = add_features(herd)
    auc_real = roc_auc_score(df["estrus"], oof_scores(df, FEATURES, "gbm", CFG))
    rng = np.random.default_rng(0)
    shuf = df.assign(estrus=df.groupby("animal_id")["estrus"]
                       .transform(lambda s: rng.permutation(s.values)))
    auc_shuf = roc_auc_score(shuf["estrus"], oof_scores(shuf, FEATURES, "gbm", CFG))
    assert auc_real > 0.85          # the planted signal is detectable
    assert abs(auc_shuf - 0.5) < 0.1  # and shuffled labels give chance
```

Add three more tests the same way: (a) event\_metrics returns recall 1.0 and zero false alerts for a perfect predictor; (b) a train/test pair sharing one animal raises in animal\_overlap; (c) evaluate.assert\_real\_only(df) raises when any row has source equal to simulated\_herd or fixture. Call assert\_real\_only at the top of train.py, which is how the PRD's FR-A3 is enforced.

Exit gate: the control test passes on your code, and fails when you deliberately make the baseline include today's value (change shift(1) to shift(0), run, then revert).

## 6. Steps 9-11: train, export, serve

**Step 9: the orchestrator and generated reports (src/train.py)**

train.py does one job: real data in, models and reports out. It aborts on simulated rows, and nothing in reports/ is typed by hand.

```python
import json, hashlib, subprocess, datetime as dt
import joblib, pandas as pd
from src.common import load_config, require, seed_all
from src.features import add_features
from src.evaluate import (assert_real_only, split_animals, event_metrics,
                          pick_threshold, bootstrap_by_animal)
from src.models.estrus import FEATURE_SETS, make_model, oof_scores, score
from sklearn.metrics import average_precision_score

RUNS = [("rule", "activity")] + [(m, f) for m in ("logistic", "gbm") for f in FEATURE_SETS]
VERSION = "estrus-1.0.0"

def main():
    cfg = load_config(); seed_all(cfg["seed"])
    reg = require("data4_eartag", "estrus_train")
    path = "data/interim/eartag_animal_day.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    assert_real_only(df)
    data_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    df = add_features(df)
    dev_ids, test_ids = split_animals(df["animal_id"].unique(), cfg["test_frac"], cfg["seed"])
    dev = df[df["animal_id"].isin(dev_ids)].reset_index(drop=True)
    test = df[df["animal_id"].isin(test_ids)].reset_index(drop=True)
    results = []
    for name, fs in RUNS:
        feats = FEATURE_SETS[fs]
        oof = oof_scores(dev, feats, name, cfg)
        thr = pick_threshold(dev, oof, cfg["max_false_alerts_per_100"], cfg["tol_days"])
        model = None if name == "rule" else make_model(name, cfg["seed"]).fit(dev[feats], dev["estrus"])
        p = score(name, model, test, feats)
        alert = p >= thr
        m = event_metrics(test, alert, cfg["tol_days"])
        ci = bootstrap_by_animal(test, alert, cfg["tol_days"], cfg["bootstrap_reps"], cfg["seed"])
        results.append({"model": name, "features": fs, "threshold": thr, **m,
                        "recall_ci": ci["event_recall"], "fa_ci": ci["false_alerts_per_100"],
                        "pr_auc": float(average_precision_score(test["estrus"], p)),
                        "prevalence": float(test["estrus"].mean()),
                        "n_test_animals": len(test_ids)})
        joblib.dump({"model": model, "name": name, "feats": feats, "threshold": thr,
                     "version": VERSION, "data_hash": data_hash},
                    f"models/estrus-{name}-{fs}.joblib")
    metrics = {
        "run": {"date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "git": subprocess.getoutput("git rev-parse --short HEAD"),
                "data_source": "real", "data_hash": data_hash, "species": reg["species"]},
        "data": {"n_animals": int(df["animal_id"].nunique()), "n_animal_days": len(df),
                 "n_events": int(df["estrus"].sum()),
                 "n_animals_with_event": int(df.loc[df["estrus"] == 1, "animal_id"].nunique())},
        "estrus": results,
    }
    json.dump(metrics, open("reports/metrics.json", "w"), indent=1)
    write_model_cards(metrics)

if __name__ == "__main__":
    main()
```

Add to config.yaml: seed, test\_frac (0.2), cv\_folds (5), cv\_repeats (3), tol\_days (1), max\_false\_alerts\_per\_100 (5), bootstrap\_reps (1000). write\_model\_cards(metrics) formats a markdown table from the same dictionary plus fixed headings: data source and species, sample counts, what the alert means, tolerance used, known limits (daily resolution, one species, one site). Behaviour and Smid results go into the same metrics.json under their own keys, written by their own train functions, so the dashboard has one source.

Exit gate: delete reports/ and models/, run python -m src.train, and every file comes back with the same numbers (fixed seed).

**Step 10: export\_web.py and dashboard rewiring**

```python
# src/export_web.py
import json

def pct(x):
    return f"{100 * x:.0f}%"

def main():
    m = json.load(open("reports/metrics.json"))
    best = max((r for r in m["estrus"] if r["model"] == "gbm" and r["features"] == "both"),
               key=lambda r: r["event_recall"])
    rule = next(r for r in m["estrus"] if r["model"] == "rule")
    src = f"reports/metrics.json · {m['run']['git']}"
    kpis = {
        "estrus_recall": {"display": f"{pct(best['event_recall'])} ({pct(best['recall_ci'][0])}-{pct(best['recall_ci'][1])})", "source": src},
        "estrus_fa": {"display": f"{best['false_alerts_per_100']:.1f} per 100 animal-days", "source": src},
        "rule_recall": {"display": pct(rule["event_recall"]), "source": src},
        "n_events": {"display": str(m["data"]["n_events"]), "source": src},
        "n_animals": {"display": str(m["data"]["n_animals"]), "source": src},
    }
    json.dump({"kpis": kpis, "run": m["run"], "estrus": m["estrus"]},
              open("web/data.json", "w"), indent=1)

if __name__ == "__main__":
    main()
```

In web/index.html, find every typed metric and replace it with a placeholder:

```bash
grep -nE "[0-9]+\.[0-9]{2,}|[0-9]+(\.[0-9]+)?%" web/index.html
```

```html
<span data-kpi="estrus_recall">loading</span>
<script>
fetch("data.json").then(r => r.json()).then(d => {
  document.querySelectorAll("[data-kpi]").forEach(el => {
    const k = d.kpis[el.dataset.kpi];
    el.textContent = k ? k.display : "n/a";
    if (k) el.title = k.source;
  });
});
async function scoreRows(rows) {
  const r = await fetch("/v1/estrus", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify({rows})});
  const j = await r.json();
  if (!r.ok) throw new Error(j.errors.join("; "));
  return j.results;
}
</script>
```

The sandbox now builds a short history per animal from its form inputs, calls scoreRows, and shows the returned score, threshold and model\_version. Any view built from simulation/ gets a visible SIMULATED banner. Exit gate: change cv\_repeats or the seed, run make all, and the dashboard numbers change with no file edited by hand.

**Step 11: a validated API (src/service.py)**

The model needs about 14 days of history per animal, so the API takes a time series, not a single day.

```python
import json, joblib, pandas as pd
from src.features import add_features
from src.models.estrus import score

REQUIRED = ["animal_id", "date", "steps_sum", "temp_mean", "temp_q90"]
LIMITS = {"max_rows": 5000,
          "ranges": {"steps_sum": (0, 200000), "temp_mean": (30, 45), "temp_q90": (30, 45)}}
BUNDLE = joblib.load("models/estrus-gbm-both.joblib")

def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        return ["rows must be a non-empty list"]
    if len(rows) > LIMITS["max_rows"]:
        return [f"at most {LIMITS['max_rows']} rows"]
    errs = []
    for i, r in enumerate(rows):
        miss = [k for k in REQUIRED if not isinstance(r, dict) or k not in r]
        if miss:
            errs.append(f"row {i}: missing {miss}"); continue
        try:
            pd.Timestamp(r["date"])
        except Exception:
            errs.append(f"row {i}: bad date")
        for k, (lo, hi) in LIMITS["ranges"].items():
            v = r[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                errs.append(f"row {i}: {k} must be a number in [{lo}, {hi}]")
    return errs

def score_estrus(rows):
    df = pd.DataFrame(rows).assign(estrus=0)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    f = add_features(df)
    f = f.merge(df[["animal_id", "date"]].drop_duplicates(), on=["animal_id", "date"])
    p = score(BUNDLE["name"], BUNDLE["model"], f, BUNDLE["feats"])
    ready = f["steps_sum_ratio"].notna().to_numpy()
    return [{"animal_id": a, "date": d.strftime("%Y-%m-%d"), "score": float(s),
             "alert": bool(ok and s >= BUNDLE["threshold"]), "enough_history": bool(ok),
             "threshold": BUNDLE["threshold"], "model_version": BUNDLE["version"],
             "data_source": "real"}
            for a, d, s, ok in zip(f["animal_id"], f["date"], p, ready)]

def dispatch(method, path, body=b""):
    if (method, path) == ("GET", "/healthz"):
        return 200, {"status": "ok", "model_version": BUNDLE["version"], "data_hash": BUNDLE["data_hash"]}
    if (method, path) == ("POST", "/v1/estrus"):
        try:
            payload = json.loads(body)
        except ValueError:
            return 422, {"errors": ["body is not valid JSON"]}
        rows = payload.get("rows") if isinstance(payload, dict) else None
        errs = validate_rows(rows)
        return (422, {"errors": errs[:20]}) if errs else (200, {"results": score_estrus(rows)})
    return 404, {"error": "not found"}
```

In web/server.py, call dispatch for every /healthz and /v1/\* request and send its status code unchanged. tests/test\_api.py then checks: a missing field gives 422 with the row number, temp\_mean of 60 gives 422, an empty list gives 422, and 20 days of valid history for one animal gives 200 with score, threshold and model\_version on every row.

## 7. Step 12: tests, Makefile, CI, final checklist

**Makefile.** The one-line form below avoids the tab-character problem on Windows editors.

```make
PY=python
.PHONY: all fetch data audit train web test serve

fetch: ; $(PY) scripts/fetch_data.py
data:  ; $(PY) -m src.ingest.data4_eartag && $(PY) -m src.ingest.smid && $(PY) -m src.ingest.data2_collar
audit: ; $(PY) -m src.audit
train: ; $(PY) -m src.train
web:   ; $(PY) -m src.export_web
test:  ; $(PY) -m pytest -q
serve: ; $(PY) web/server.py --port 8080
all: fetch data audit train web test
```

If make is not installed on Windows, keep the same targets as a scripts/run\_all.ps1 that runs the commands in this order.

**Tests that must not need raw data.** CI cannot hold 8 GB, so every test runs on the planted-signal fixture. For the API test, let service.py load its model from an environment variable (default models/estrus-gbm-both.joblib) and have conftest.py train a tiny gbm on the fixture and point the variable at it.

| Test file | What it proves |
| --- | --- |
| test\_features\_causal.py | Changing values after day t never changes day-t features |
| test\_leakage.py | Animal overlap raises; assert\_real\_only rejects fixture and simulated rows; duplicate animal-days raise |
| test\_controls.py | Planted signal found (AUC above 0.85); within-animal shuffle gives chance |
| test\_evaluate.py | Perfect predictor gives recall 1.0 and zero false alerts; all-alert gives recall 1.0 and many false alerts |
| test\_api.py | 422 for missing field, out-of-range value, empty list; 200 with model\_version for valid history |
| test\_determinism.py | Two runs with the same seed give identical scores |

**CI (.github/workflows/ci.yml)**

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt pytest
      - run: python -m pytest -q
```

**Final checklist**

- [ ] git history shows v1-baseline, reproduced v1 numbers, audit, honest evaluation, restructure, live dashboard
- [ ] data/registry.yaml complete; licences and species filled in
- [ ] reports/audit\_data4.md states the corrected ear-tag numbers with counts
- [ ] metrics.json shows 7 estrus rows with intervals; decision rule applied in the model card
- [ ] grep for typed metrics in web/index.html returns nothing
- [ ] Sandbox network call to /v1/estrus works; invalid input returns 422
- [ ] SIMULATED banner on every view built from simulation/
- [ ] Fresh clone: pip install, make all, make serve works from the README alone
- [ ] CI shows a green check
- [ ] Tag v2.0 and export the README results table into your report

**Suggested order against the six-week plan:** steps 0-2 in week 1, steps 3-4 in weeks 1-2, steps 5-8 in weeks 2-3, step 9 and the restructure in weeks 3-4, steps 10-11 in weeks 4-5, step 12 in weeks 5-6.
