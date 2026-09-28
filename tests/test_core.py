"""Kiểm thử các hàm lõi (không cần GPU, không cần tải model). Chạy: python tests/test_core.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import nltk  # noqa: E402

for _res in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

from mcplus import DATASETS_11, PAPER_TABLE2  # noqa: E402
from mcplus.aggregator import balanced_weights, cv_select, fit_predict, leave_one_dataset_out  # noqa: E402
from mcplus.chunking import make_chunks, sent_tokenize_with_newlines, split_claim  # noqa: E402
from mcplus.evidence import select_evidence  # noqa: E402
from mcplus.features import build_features, feature_sets, lexical_overlap  # noqa: E402
from mcplus.metrics import paired_bootstrap, per_dataset, rates, tune_threshold  # noqa: E402
from mcplus.report import fill_template, md_table  # noqa: E402


def minicheck_reference_chunks(doc, n):
    """Chép nguyên văn nhánh Flan-T5 của hàm chunks() trong MiniCheck (inference.py @ b58b9fa)."""
    def chunks(lst, n):
        current_chunk = []
        current_word_count = 0
        for sentence in lst:
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count > n:
                yield ' '.join(current_chunk)
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        if current_chunk:
            yield ' '.join(current_chunk)

    doc_sents = sent_tokenize_with_newlines(doc)
    doc_sents = doc_sents or ['']
    doc_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(doc_sents, n)]
    return [chunk for chunk in doc_chunks if chunk != '']


def test_chunking_matches_minicheck():
    rng = np.random.RandomState(0)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    for _ in range(200):
        sents = []
        for _ in range(rng.randint(1, 30)):
            s = " ".join(rng.choice(words, rng.randint(1, 40))).capitalize() + "."
            sents.append(s + ("\n" if rng.rand() < 0.2 else " "))
        doc = "".join(sents).strip()
        n = int(rng.choice([5, 20, 50, 500]))
        ours = make_chunks(doc, n, lambda s: len(s.split()))
        assert ours == minicheck_reference_chunks(doc, n), (doc, n)


def test_split_claim():
    assert split_claim("One fact here. Another fact there.") == ["One fact here.", "Another fact there."]
    assert split_claim("Only one.") == ["Only one."]


def test_select_evidence():
    sents = [f"s{i}" for i in range(10)]
    lengths = [10] * 10
    ranking = [7, 2, 9, 0, 1, 3, 4, 5, 6, 8]
    keep = select_evidence(sents, ranking, lengths, budget=40, top_k=None, window=1)
    assert keep == sorted(keep) and sum(lengths[i] for i in keep) <= 40
    assert {6, 7, 8}.issubset(keep)  # hạt giống đầu + 2 lân cận
    keep_k1 = select_evidence(sents, ranking, lengths, budget=100, top_k=1, window=0)
    assert keep_k1 == [7]
    long = select_evidence(sents, ranking, [100] * 10, budget=40, top_k=None, window=0)
    assert long == [7]  # câu tốt nhất luôn được giữ dù dài hơn ngân sách


def test_rates_and_threshold():
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 500)
    yhat = rng.randint(0, 2, 500)
    assert abs(rates(y, yhat)["bacc"] - balanced_accuracy_score(y, yhat)) < 1e-12
    scores = np.where(y == 1, rng.uniform(0.3, 1.0, 500), rng.uniform(0.0, 0.3, 500))
    t = tune_threshold(scores, y)
    assert rates(y, (scores > t).astype(int))["bacc"] == 1.0


def _fake_frame(n_per=60, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for d in DATASETS_11:
        for i in range(n_per):
            label = int(rng.rand() < 0.6)
            n_chunks = rng.randint(1, 5)
            probs = np.clip(rng.normal(0.35 + 0.3 * label, 0.2, n_chunks), 0.001, 0.999)
            rows.append({"uid": f"x:{d}:{i}", "dataset": d, "label": label, "chunk_probs": json.dumps([float(p) for p in probs]),
                         "n_chunks": n_chunks, "p_max": probs.max(), "doc_len": rng.randint(50, 2000),
                         "claim_len": rng.randint(5, 60), "n_claim_sents": rng.randint(1, 3),
                         "overlap": rng.rand(), "e_score": np.clip(probs.max() + rng.normal(0, 0.1), 0.001, 0.999),
                         "e_split": rng.rand()})
    return pd.DataFrame(rows)


def test_bootstrap_identity_and_direction():
    df = _fake_frame()
    df["a"] = (df["p_max"] > 0.5).astype(int)
    r = paired_bootstrap(df, "a", "a", n_boot=500)
    assert r["diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    df["oracle"] = df["label"]
    r2 = paired_bootstrap(df, "a", "oracle", n_boot=500)
    assert r2["diff"] > 0 and r2["ci_low"] > 0 and r2["p_value"] < 0.01


def test_features_and_aggregator():
    dev, test = _fake_frame(seed=2), _fake_frame(seed=3)
    dev = pd.concat([dev, build_features(dev)], axis=1)
    test = pd.concat([test, build_features(test)], axis=1)
    sets = feature_sets(dev.columns)
    assert "all+split" in sets and not dev[sets["all+split"]].isna().any().any()
    w = balanced_weights(dev["dataset"].values, dev["label"].values)
    cell_sums = pd.Series(w).groupby([dev["dataset"].values, dev["label"].values]).sum()
    assert np.allclose(cell_sums.values, cell_sums.values[0]) and abs(w.mean() - 1) < 1e-9
    params, table, oof = cv_select("lr", dev, sets["chunk_only"], n_splits=3)
    assert len(table) == 4 and oof.shape == (len(dev),)
    p = fit_predict("hgb", {"max_depth": 2, "learning_rate": 0.1, "max_iter": 50}, dev, test, sets["all"])
    assert p.shape == (len(test),) and ((p >= 0) & (p <= 1)).all()
    lodo = leave_one_dataset_out("lr", params, dev, test, sets["all"])
    assert not np.isnan(lodo).any()


def test_overlap():
    assert lexical_overlap("The cat sat on the mat", "the cat sat") == 1.0
    assert lexical_overlap("The cat sat", "dogs bark loudly") == 0.0


def test_report_helpers():
    df = pd.DataFrame({"x": [1.0, 2.5], "y": [3.0, 0.5]}, index=["r1", "r2"])
    md = md_table(df, bold_max_rows="all")
    assert "**3.00**" in md and "**2.50**" in md
    text = "a <!--AUTO:t-->old<!--/AUTO:t--> b"
    assert fill_template(text, {"t": "new"}) == "a <!--AUTO:t-->\nnew\n<!--/AUTO:t--> b"


def test_paper_table_average():
    for model in ("deberta-v3-large", "flan-t5-large"):
        vals = [v for k, v in PAPER_TABLE2[model].items() if k != "avg"]
        assert len(vals) == 10 and abs(np.mean(vals) - PAPER_TABLE2[model]["avg"]) < 0.1


def test_build_notebooks_parser():
    from build_notebooks import parse

    cells = parse("# %% [markdown]\n# # Title\n# text\n\n# %%\nx = 1\n\n# %%\nprint(x)\n")
    assert [k for k, _ in cells] == ["markdown", "code", "code"]
    assert cells[0][1] == "# Title\ntext"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok ", name)
    print(f"{len(tests)} tests passed")
