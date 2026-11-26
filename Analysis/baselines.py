"""Supervised baselines: TF-IDF + LR, GIST-embedding LR probe, and fine-tuned DistilBERT.

All three train on the DiFrauD train split, preprocessed as the knowledge base is (documents
under 5 words dropped, longer ones truncated to 900 words; 76,311 documents), and predict the
3,000 evaluation texts in Outputs/total_samples.csv. No LLM is involved, and every input gets a label.

    tfidf       unigrams + bigrams (min_df 5, 50,000 features), class-balanced logistic regression
    gist        class-balanced logistic regression over standardized, unnormalized
                avsolatorio/GIST-small-Embedding-v0 embeddings of the unchunked documents
    distilbert  distilbert-base-uncased, 2 epochs, lr 2e-5, batch 32, 512 tokens, seed 42,
                class-weighted cross-entropy, trained on 95% of the train split; the other 5%
                is only logged after each epoch (no checkpoint selection)

    uv run python Analysis/baselines.py                       # tfidf and gist (CPU is enough)
    uv run python Analysis/baselines.py --only distilbert     # needs a GPU; about 20-60 min
    uv run python Analysis/baselines.py --only tfidf gist distilbert

DiFrauD is downloaded on first use into Saved/datasets/ (the same cache main.py uses); GIST
embeddings and the fine-tuned model are cached under Saved/baselines/.

Outputs (Outputs/analysis/baselines/):
    baseline_<name>_outputs.csv   one row per evaluation text: id, domain, gold, prediction, P(deceptive)
    baseline_metrics_summary.csv  per-domain, pooled and macro-domain F1, accuracy, MCC, balanced accuracy
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from common import ANALYSIS_DIR, DOMAINS, ROOT, SAMPLES

SEED = 42
SAVED_DIR = ROOT / "Saved" / "baselines"
DIFRAUD_REPO = "redasers/difraud"
DIFRAUD_LOCAL = ROOT / "Saved" / "datasets" / "redasers_difraud" / "main"
DIFRAUD_DOMAINS = {"fake_news": "news", "phishing": "email", "sms": "sms", "twitter_rumours": "twitter",
                   "job_scams": "message boards", "product_reviews": "product reviews",
                   "political_statements": "political statements"}
MIN_WORDS, MAX_WORDS = 5, 900

GIST_MODEL = "avsolatorio/GIST-small-Embedding-v0"
GIST_BATCH = 512

FT_MODEL = "distilbert-base-uncased"
FT_BATCH, FT_EPOCHS, FT_LR, FT_MAX_LEN = 32, 2, 2e-5, 512
FT_HOLDOUT_FRAC = 0.05
FT_PREDICT_BATCH = 64


# -- data -------------------------------------------------------------------------------------
def load_train() -> pd.DataFrame:
    """DiFrauD train split (text, label, domain) after the knowledge-base length preprocessing."""
    from huggingface_hub import hf_hub_download

    frames = []
    for hf_domain, domain in DIFRAUD_DOMAINS.items():
        local = DIFRAUD_LOCAL / hf_domain / "train.jsonl"
        if not local.exists():
            remote = hf_hub_download(repo_id=DIFRAUD_REPO, repo_type="dataset", filename=f"{hf_domain}/train.jsonl")
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote, local)
        df = pd.read_json(local, lines=True)
        df["domain"] = domain
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    words = out["text"].astype(str).str.split()
    keep = words.str.len() >= MIN_WORDS
    out = out[keep].copy()
    out["text"] = words[keep].str[:MAX_WORDS].str.join(" ")
    out["label"] = out["label"].astype(int)
    out = out.reset_index(drop=True)
    print(f"DiFrauD train: {len(out)} documents ({int((~keep).sum())} under {MIN_WORDS} words dropped), "
          f"{out['label'].mean():.1%} deceptive")
    return out


def load_eval() -> pd.DataFrame:
    ev = pd.read_csv(SAMPLES).rename(columns={"total_samples": "sample_run"})
    ev["label"] = ev["label"].astype(int)
    return ev


# -- metrics and output -----------------------------------------------------------------------
def _mcc(y, p) -> float:
    if pd.Series(y).nunique() < 2 or pd.Series(p).nunique() < 2:
        return float("nan")
    return float(matthews_corrcoef(y, p))


def _row(name, domain, d, f1=None) -> dict:
    y, p = d["true_label"], d["pred_label"]
    return {"baseline": name, "domain": domain, "n": len(d), "n_pos": int(y.sum()),
            "f1": round(f1 if f1 is not None else f1_score(y, p, average="binary", pos_label=1, zero_division=0), 4),
            "accuracy": round(float(accuracy_score(y, p)), 4), "mcc": round(_mcc(y, p), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y, p)), 4)}


def metrics(pred: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-domain rows, then pooled (domains pooled) and macro (mean of per-domain F1) rows."""
    rows = [_row(name, dom, pred[pred["domain"] == dom]) for dom in DOMAINS]
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    rows.append(_row(name, "ALL (pooled)", pred))
    rows.append(_row(name, "ALL (macro)", pred, f1=macro_f1))
    return pd.DataFrame(rows)


def write_outputs(name: str, ev: pd.DataFrame, pred, prob, out_dir: Path) -> pd.DataFrame:
    out = pd.DataFrame({"baseline": name, "id": ev["id"], "domain": ev["domain"], "sample_run": ev["sample_run"],
                        "true_label": ev["label"], "pred_label": np.asarray(pred).astype(int),
                        "prob_deceptive": np.round(np.asarray(prob, dtype=float), 6)})
    out["valid"] = True
    out["correct"] = out["true_label"] == out["pred_label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"baseline_{name}_outputs.csv", index=False)
    m = metrics(out, name)
    print(m.to_string(index=False))
    return m


# -- baselines --------------------------------------------------------------------------------
def run_tfidf(train, ev):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50_000)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    x = vec.transform(ev["content"].astype(str))
    return "tfidf_lr", clf.predict(x), clf.predict_proba(x)[:, 1]


def _embed(texts: list[str], cache_name: str) -> np.ndarray:
    """GIST-small embeddings, cached under Saved/baselines/ by a hash of the input texts."""
    import torch
    from sentence_transformers import SentenceTransformer

    key = hashlib.md5("\x1f".join(texts).encode("utf-8", errors="replace")).hexdigest()[:12]
    cache = SAVED_DIR / f"{cache_name}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(texts)} texts with {GIST_MODEL} on {device}")
    model = SentenceTransformer(GIST_MODEL, device=device)
    with torch.inference_mode():
        emb = model.encode(texts, batch_size=GIST_BATCH if device == "cuda" else 64, show_progress_bar=True,
                           normalize_embeddings=False, convert_to_numpy=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb.astype(np.float32))
    return emb


def run_gist(train, ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = _embed(train["text"].tolist(), "gist_train")
    x_eval = _embed(ev["content"].astype(str).tolist(), "gist_eval")
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(scaler.transform(x_train), train["label"])
    x = scaler.transform(x_eval)
    return "gist_lr", clf.predict(x), clf.predict_proba(x)[:, 1]


def run_distilbert(train, ev, retrain: bool = False):
    import torch
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts, self.labels = list(texts), list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            return {"text": self.texts[i], "labels": self.labels[i]}

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, **kwargs):
            super().__init__(**kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels,
                                                     weight=self.class_weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], truncation=True, max_length=FT_MAX_LEN, padding=True,
                        return_tensors="pt")
        enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return enc

    def holdout_metrics(eval_pred):
        preds, labels = eval_pred.predictions.argmax(axis=-1), eval_pred.label_ids
        return {"deceptive_f1": f1_score(labels, preds, average="binary", pos_label=1, zero_division=0),
                "accuracy": float((preds == labels).mean())}

    set_seed(SEED)
    save_dir = SAVED_DIR / "finetuned_distilbert"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if save_dir.exists() and not retrain:
        print(f"Loading the fine-tuned model from {save_dir} (pass --retrain to train again)")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    else:
        tr_x, ho_x, tr_y, ho_y = train_test_split(train["text"].tolist(), train["label"].tolist(),
                                                  test_size=FT_HOLDOUT_FRAC, stratify=train["label"],
                                                  random_state=SEED)
        counts = np.bincount(tr_y, minlength=2)
        class_weights = torch.tensor(len(tr_y) / (2.0 * counts), dtype=torch.float)
        tokenizer = AutoTokenizer.from_pretrained(FT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FT_MODEL, num_labels=2)
        args = TrainingArguments(
            output_dir=str(SAVED_DIR / "finetuned_distilbert_ckpt"),
            num_train_epochs=FT_EPOCHS, learning_rate=FT_LR,
            per_device_train_batch_size=FT_BATCH, per_device_eval_batch_size=FT_BATCH * 2,
            eval_strategy="epoch", save_strategy="no", logging_steps=100,
            remove_unused_columns=False,  # the collator needs the raw "text" field
            bf16=torch.cuda.is_available(), dataloader_num_workers=0, report_to=[], seed=SEED)
        trainer = WeightedTrainer(class_weights=class_weights, model=model, args=args,
                                  train_dataset=TextDataset(tr_x, tr_y), eval_dataset=TextDataset(ho_x, ho_y),
                                  data_collator=collate, processing_class=tokenizer, compute_metrics=holdout_metrics)
        trainer.train()
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    texts = ev["content"].astype(str).tolist()
    model.to(device).eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), FT_PREDICT_BATCH):
            enc = tokenizer(texts[i:i + FT_PREDICT_BATCH], truncation=True, max_length=FT_MAX_LEN, padding=True,
                            return_tensors="pt").to(device)
            probs.append(torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy())
    prob = np.concatenate(probs)
    return "finetuned_distilbert", (prob >= 0.5).astype(int), prob


RUNNERS = {"tfidf": run_tfidf, "gist": run_gist, "distilbert": run_distilbert}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(RUNNERS), default=["tfidf", "gist"])
    ap.add_argument("--retrain", action="store_true", help="retrain DistilBERT even if a saved model exists")
    ap.add_argument("--out", type=Path, default=ANALYSIS_DIR / "baselines")
    a = ap.parse_args()

    train, ev = load_train(), load_eval()
    for key in a.only:
        if key == "distilbert":
            name, pred, prob = run_distilbert(train, ev, retrain=a.retrain)
        else:
            name, pred, prob = RUNNERS[key](train, ev)
        write_outputs(name, ev, pred, prob, a.out)

    frames = [metrics(pd.read_csv(p), p.stem.removeprefix("baseline_").removesuffix("_outputs"))
              for p in sorted(a.out.glob("baseline_*_outputs.csv"))]
    pd.concat(frames, ignore_index=True).to_csv(a.out / "baseline_metrics_summary.csv", index=False)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
