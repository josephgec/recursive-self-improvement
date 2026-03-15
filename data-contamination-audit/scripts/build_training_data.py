#!/usr/bin/env python
"""Build labelled training data for the human-vs-synthetic classifier.

Sources
-------
A. HC3 dataset (Hello-SimpleAI/HC3) — human answers (label=0) and ChatGPT
   answers (label=1).
B. GPT-2 synthetic text generation — diverse seed prompts, label=1.
C. Pre-2020 Common Crawl (CC-MAIN-2017-22, CC-MAIN-2019-22) — label=0.

Usage
-----
    python scripts/build_training_data.py
    python scripts/build_training_data.py --n-human 1000 --n-synthetic 1000
    python scripts/build_training_data.py --output-dir data/training_v2/

NOTE: Imports are carefully ordered to avoid a known segfault on macOS ARM64
where loading xgboost/sklearn before torch GPT-2 inference causes a crash.
Feature extraction (torch-based) is performed *before* any sklearn/xgboost
imports.
"""

from __future__ import annotations

import hashlib
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.common_crawl import Document

# ---------------------------------------------------------------------------
# Logging & CLI
# ---------------------------------------------------------------------------

console = Console()
logger = logging.getLogger("build_training_data")

app = typer.Typer(help="Build labelled training data for the contamination classifier.")

MIN_CHAR_LENGTH = 100

# ---------------------------------------------------------------------------
# Seed prompts for GPT-2 generation (50+ diverse prompts)
# ---------------------------------------------------------------------------

_GPT2_SEED_PROMPTS: list[str] = [
    # Science
    "Recent advances in quantum computing have demonstrated that",
    "The theory of general relativity predicts that",
    "Researchers at the institute published new findings showing",
    "In a groundbreaking study on CRISPR gene editing, scientists found",
    "The origins of the universe can be traced back to",
    "New observations from the James Webb Space Telescope reveal",
    "The periodic table of elements was originally organized by",
    "Climate models have consistently predicted that global temperatures",
    # Technology
    "The evolution of artificial intelligence over the past decade has",
    "Modern web development frameworks such as React and Angular have",
    "Cloud computing infrastructure has fundamentally changed how",
    "Cybersecurity experts warn that the increasing sophistication of",
    "The rise of blockchain technology has enabled new forms of",
    "Open source software development has transformed the way",
    "Machine learning algorithms are now capable of",
    "The semiconductor industry faces unprecedented challenges as",
    # News & Current Events
    "In a statement released earlier today, the committee announced",
    "Analysts predict that the global economy will experience",
    "The international summit concluded with a joint declaration on",
    "Public health officials have issued new guidelines regarding",
    "The latest quarterly earnings report showed that the company",
    "Diplomatic relations between the two nations have been strained by",
    # History
    "The Industrial Revolution began in the late 18th century when",
    "Ancient civilizations along the Nile River developed sophisticated",
    "The fall of the Roman Empire was caused by a combination of",
    "During the Renaissance period, European artists and scholars",
    "The discovery of the Americas by European explorers led to",
    "World War II fundamentally reshaped the geopolitical landscape",
    "The French Revolution of 1789 was sparked by widespread",
    "The Silk Road facilitated trade and cultural exchange between",
    # Health & Medicine
    "Medical researchers have identified a new biomarker for early detection of",
    "The human immune system responds to viral infections by",
    "Nutritional science has evolved significantly in recent years, with",
    "Mental health awareness campaigns have helped reduce the stigma of",
    "Advances in surgical robotics have enabled doctors to perform",
    "The development of mRNA vaccine technology was accelerated by",
    "Studies on the gut microbiome suggest that intestinal bacteria",
    "Sleep research has shown that chronic sleep deprivation leads to",
    # Education
    "The modern education system was largely shaped by reforms in the",
    "Online learning platforms have made it possible for students to",
    "Research in cognitive science suggests that effective studying requires",
    "Universities around the world are adapting their curricula to",
    # Environment
    "Deforestation in tropical regions continues to threaten biodiversity by",
    "Renewable energy sources such as solar and wind power have become",
    "Ocean acidification caused by increased carbon dioxide absorption",
    "The global water crisis is expected to intensify as populations grow",
    "Conservation efforts in national parks have successfully restored",
    # Economics & Business
    "Supply chain disruptions have highlighted the fragility of",
    "The gig economy has transformed traditional employment models by",
    "Central banks around the world have responded to inflation by",
    "Venture capital investment in startups reached record levels when",
    "International trade agreements have shaped global commerce by",
    # Philosophy & Society
    "The concept of social justice has evolved throughout history to",
    "Ethical debates surrounding artificial intelligence center on",
    "The philosophy of science examines the foundations and methods of",
]


# ---------------------------------------------------------------------------
# Source A: HC3 dataset
# ---------------------------------------------------------------------------


def _load_hc3(
    n_human: int,
    n_synthetic: int,
    seed: int,
) -> tuple[list[Document], list[Document]]:
    """Load human and ChatGPT answers from the HC3 dataset.

    Returns
    -------
    (human_docs, synthetic_docs)
    """
    import datasets  # noqa: E402  (deferred to keep top-level lightweight)

    console.print("[bold cyan]Loading HC3 dataset from HuggingFace…[/bold cyan]")
    ds = datasets.load_dataset("Hello-SimpleAI/HC3", "all")

    # HC3 has a single split called "train".
    split = ds["train"]

    rng = random.Random(seed)

    human_texts: list[str] = []
    synthetic_texts: list[str] = []

    for row in split:
        # Each row has 'human_answers' and 'chatgpt_answers' — both are lists.
        for ans in row.get("human_answers", []):
            if isinstance(ans, str) and len(ans) >= MIN_CHAR_LENGTH:
                human_texts.append(ans)
        for ans in row.get("chatgpt_answers", []):
            if isinstance(ans, str) and len(ans) >= MIN_CHAR_LENGTH:
                synthetic_texts.append(ans)

    rng.shuffle(human_texts)
    rng.shuffle(synthetic_texts)

    human_texts = human_texts[:n_human]
    synthetic_texts = synthetic_texts[:n_synthetic]

    now = datetime.now()

    human_docs: list[Document] = []
    for i, text in enumerate(human_texts):
        doc_id = hashlib.sha256(
            f"hc3_human:{i}:{text[:64]}".encode()
        ).hexdigest()[:16]
        human_docs.append(
            Document(
                doc_id=f"hc3h-{doc_id}",
                text=text,
                source="hc3_human",
                timestamp=now,
                url=None,
                metadata={"dataset": "HC3", "answer_type": "human"},
            )
        )

    synthetic_docs: list[Document] = []
    for i, text in enumerate(synthetic_texts):
        doc_id = hashlib.sha256(
            f"hc3_chatgpt:{i}:{text[:64]}".encode()
        ).hexdigest()[:16]
        synthetic_docs.append(
            Document(
                doc_id=f"hc3s-{doc_id}",
                text=text,
                source="hc3_chatgpt",
                timestamp=now,
                url=None,
                metadata={"dataset": "HC3", "answer_type": "chatgpt"},
            )
        )

    console.print(
        f"  [green]HC3: {len(human_docs)} human, "
        f"{len(synthetic_docs)} synthetic answers[/green]"
    )
    return human_docs, synthetic_docs


# ---------------------------------------------------------------------------
# Source B: GPT-2 generated text
# ---------------------------------------------------------------------------


def _generate_gpt2_texts(
    n: int,
    seed: int,
) -> list[Document]:
    """Generate synthetic text using GPT-2 and diverse seed prompts."""
    if n <= 0:
        return []

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    console.print(
        f"[bold cyan]Generating {n} synthetic texts with GPT-2…[/bold cyan]"
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    # Set pad_token to eos_token to avoid warnings during generation.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = random.Random(seed)
    prompts = _GPT2_SEED_PROMPTS.copy()
    now = datetime.now()

    documents: list[Document] = []

    from rich.progress import Progress

    with Progress(console=console) as progress:
        task = progress.add_task("GPT-2 generation", total=n)

        prompt_idx = 0
        while len(documents) < n:
            # Cycle through prompts, shuffling each pass.
            if prompt_idx >= len(prompts):
                rng.shuffle(prompts)
                prompt_idx = 0

            prompt = prompts[prompt_idx]
            prompt_idx += 1

            inputs = tokenizer(prompt, return_tensors="pt")
            with __import__("torch").no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    top_k=50,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            if len(generated_text) < MIN_CHAR_LENGTH:
                continue

            doc_id = hashlib.sha256(
                f"gpt2:{len(documents)}:{generated_text[:64]}".encode()
            ).hexdigest()[:16]

            documents.append(
                Document(
                    doc_id=f"gpt2-{doc_id}",
                    text=generated_text,
                    source="gpt2_generated",
                    timestamp=now,
                    url=None,
                    metadata={
                        "generator": "gpt2",
                        "prompt": prompt,
                        "top_k": 50,
                        "temperature": 0.8,
                    },
                )
            )
            progress.advance(task)

    console.print(
        f"  [green]GPT-2: generated {len(documents)} synthetic documents[/green]"
    )
    return documents


# ---------------------------------------------------------------------------
# Source C: Pre-2020 Common Crawl
# ---------------------------------------------------------------------------

_PRE2020_CRAWL_IDS = ["CC-MAIN-2017-22", "CC-MAIN-2019-22"]


def _load_common_crawl_human(n: int, seed: int) -> list[Document]:
    """Sample human-written text from pre-2020 Common Crawl archives."""
    if n <= 0:
        return []

    from src.data.common_crawl import sample_cc_warc

    console.print(
        f"[bold cyan]Sampling {n} documents from pre-2020 Common Crawl…[/bold cyan]"
    )

    per_crawl = max(1, n // len(_PRE2020_CRAWL_IDS))
    remainder = n - per_crawl * len(_PRE2020_CRAWL_IDS)

    documents: list[Document] = []

    for i, crawl_id in enumerate(_PRE2020_CRAWL_IDS):
        target = per_crawl + (remainder if i == len(_PRE2020_CRAWL_IDS) - 1 else 0)
        logger.info("Sampling %d docs from %s", target, crawl_id)

        try:
            docs = sample_cc_warc(
                crawl_id=crawl_id,
                n=target,
                seed=seed + i,
                languages=["en"],
            )
        except Exception:
            logger.warning(
                "Failed to sample from %s — skipping this crawl.",
                crawl_id,
                exc_info=True,
            )
            continue

        # Re-label source and filter length.
        for doc in docs:
            doc.source = "cc_pre2020"
            doc.metadata["original_source"] = "common_crawl"
        docs = [d for d in docs if len(d.text) >= MIN_CHAR_LENGTH]
        documents.extend(docs)

    console.print(
        f"  [green]Common Crawl: sampled {len(documents)} human documents[/green]"
    )
    return documents


# ---------------------------------------------------------------------------
# Feature extraction (torch-first to avoid segfault)
# ---------------------------------------------------------------------------


def _extract_features(
    documents: list[Document],
    output_dir: Path,
) -> None:
    """Run feature extraction and save outputs.

    CRITICAL: This function imports PerplexityScorer / build_feature_matrix
    (which load torch and GPT-2) BEFORE any xgboost or sklearn imports to
    avoid the macOS ARM64 segfault.
    """
    # --- torch-based imports FIRST ---
    from src.classifier.features.ensemble import build_feature_matrix
    from src.classifier.features.perplexity import PerplexityScorer
    from src.classifier.features.watermark import WatermarkDetector

    console.print("[bold cyan]Extracting features…[/bold cyan]")

    perplexity_scorer = PerplexityScorer(model_name="gpt2")
    watermark_detector = WatermarkDetector()
    tokenizer = perplexity_scorer.tokenizer

    cache_dir = output_dir / "features_cache"
    feature_matrix = build_feature_matrix(
        documents=documents,
        perplexity_scorer=perplexity_scorer,
        watermark_detector=watermark_detector,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
    )

    features_path = output_dir / "features.parquet"
    feature_matrix.to_parquet(features_path, index=False)
    logger.info("Saved feature matrix (%s) to %s", feature_matrix.shape, features_path)

    console.print(
        f"  [green]Features: {feature_matrix.shape[1] - 1} features "
        f"for {feature_matrix.shape[0]} documents → {features_path}[/green]"
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _save_labels(
    documents: list[Document],
    labels: list[int],
    output_dir: Path,
) -> None:
    """Save doc_id, label, source to a CSV file."""
    import pandas as pd

    rows = []
    for doc, label in zip(documents, labels):
        rows.append(
            {"doc_id": doc.doc_id, "label": label, "source": doc.source}
        )

    df = pd.DataFrame(rows)
    labels_path = output_dir / "labels.csv"
    df.to_csv(labels_path, index=False)
    logger.info("Saved %d labels to %s", len(df), labels_path)
    console.print(f"  [green]Labels: {len(df)} entries → {labels_path}[/green]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def main(
    n_human: int = typer.Option(
        500,
        "--n-human",
        help="Total number of human-written documents to collect.",
    ),
    n_synthetic: int = typer.Option(
        500,
        "--n-synthetic",
        help="Total number of synthetic/AI-generated documents to collect.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for training data. Defaults to data/training/.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Build labelled training data for the human-vs-synthetic classifier."""
    _setup_logging(verbose)

    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "training"
    elif not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Build Training Data")
    console.print(f"[bold]Target:[/bold] {n_human} human + {n_synthetic} synthetic")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    console.print(f"[bold]Seed:[/bold] {seed}")
    console.print()

    # Half-quotas for HC3.
    hc3_human_quota = n_human // 2
    hc3_synthetic_quota = n_synthetic // 2

    all_human_docs: list[Document] = []
    all_synthetic_docs: list[Document] = []

    # ------------------------------------------------------------------
    # Source A: HC3
    # ------------------------------------------------------------------
    try:
        hc3_human, hc3_synthetic = _load_hc3(
            n_human=hc3_human_quota,
            n_synthetic=hc3_synthetic_quota,
            seed=seed,
        )
        all_human_docs.extend(hc3_human)
        all_synthetic_docs.extend(hc3_synthetic)
    except Exception:
        logger.warning(
            "HC3 dataset unavailable — falling back to GPT-2 + CC only.",
            exc_info=True,
        )
        console.print(
            "[yellow]HC3 dataset unavailable. "
            "Will fill quotas from GPT-2 and Common Crawl.[/yellow]"
        )

    # ------------------------------------------------------------------
    # Source B: GPT-2 synthetic generation (fill remaining synthetic quota)
    # ------------------------------------------------------------------
    remaining_synthetic = n_synthetic - len(all_synthetic_docs)
    try:
        gpt2_docs = _generate_gpt2_texts(n=remaining_synthetic, seed=seed)
        all_synthetic_docs.extend(gpt2_docs)
    except Exception:
        logger.warning(
            "GPT-2 generation failed.", exc_info=True,
        )
        console.print("[yellow]GPT-2 generation failed.[/yellow]")

    # ------------------------------------------------------------------
    # Source C: Pre-2020 Common Crawl (fill remaining human quota)
    # ------------------------------------------------------------------
    remaining_human = n_human - len(all_human_docs)
    try:
        cc_docs = _load_common_crawl_human(n=remaining_human, seed=seed)
        all_human_docs.extend(cc_docs)
    except Exception:
        logger.warning(
            "Common Crawl sampling failed — using only HC3 human data.",
            exc_info=True,
        )
        console.print(
            "[yellow]Common Crawl unavailable. "
            "Human data limited to HC3.[/yellow]"
        )

    # ------------------------------------------------------------------
    # Combine and shuffle
    # ------------------------------------------------------------------
    human_labels = [0] * len(all_human_docs)
    synthetic_labels = [1] * len(all_synthetic_docs)

    all_docs = all_human_docs + all_synthetic_docs
    all_labels = human_labels + synthetic_labels

    # Shuffle together.
    rng = random.Random(seed)
    combined = list(zip(all_docs, all_labels))
    rng.shuffle(combined)
    all_docs = [d for d, _ in combined]
    all_labels = [l for _, l in combined]

    console.print()
    console.rule("[bold blue]Summary")
    console.print(f"  Human documents:    {len(all_human_docs)}")
    console.print(f"  Synthetic documents: {len(all_synthetic_docs)}")
    console.print(f"  Total:              {len(all_docs)}")
    console.print()

    if not all_docs:
        console.print("[red]No documents collected — aborting.[/red]")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Feature extraction (torch-first — BEFORE any sklearn/xgboost)
    # ------------------------------------------------------------------
    _extract_features(all_docs, output_dir)

    # ------------------------------------------------------------------
    # Save labels
    # ------------------------------------------------------------------
    _save_labels(all_docs, all_labels, output_dir)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    console.print()
    console.rule("[bold green]Training data build complete")
    console.print(f"  Features → {output_dir / 'features.parquet'}")
    console.print(f"  Labels   → {output_dir / 'labels.csv'}")


if __name__ == "__main__":
    app()
