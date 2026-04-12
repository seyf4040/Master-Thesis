#!/usr/bin/env python3
"""
generate_synthetic_data.py — Synthetic French training data generation (Phase 2, Track A)

Generates synthetic training examples for under-represented HateCheck-FR functionalities
using a locally-deployed instruction-tuned LLM (default: Mistral-7B-Instruct).

Each generated example is validated with:
  1. Language detection (must be French, via langdetect)
  2. Length filter (10–150 tokens)
  3. Exact deduplication (within and across runs via seen-hashes file)

Output JSONL fields: text, label, functionality, source, model, timestamp

Functionalities (label):
  slur_h          (1) — explicit slur targeting identity group
  spell_leet_h    (1) — leet-speak obfuscation (h@te, s@l0p, …)
  spell_char_del_h(1) — character-deletion obfuscation (hte, slp, …)
  derog_impl_h    (1) — implicit derogation, no explicit slur
  counter_quote_nh(0) — counter-speech that quotes hate to condemn it

Usage:
    python code/generate_synthetic_data.py \\
        --functionality slur_h \\
        --n_samples 300 \\
        --output_dir ~/code/data/synthetic \\
        --model mistral_7b_instruct \\
        --batch_size 10 \\
        --cache_dir ~/datasets/cache

    # All functionalities at once:
    python code/generate_synthetic_data.py \\
        --functionality all \\
        --n_samples 300 \\
        --output_dir ~/code/data/synthetic \\
        --model mistral_7b_instruct
"""

import gc
import json
import time
import hashlib
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────────────────

MODELS: Dict[str, str] = {
    "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral_7b_instruct_v2": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

DEFAULT_MODEL = "mistral_7b_instruct"


# ── Functionality config ──────────────────────────────────────────────────────

FUNCTIONALITIES = {
    "slur_h": {"label": 1},
    "spell_leet_h": {"label": 1},
    "spell_char_del_h": {"label": 1},
    "derog_impl_h": {"label": 1},
    "counter_quote_nh": {"label": 0},
}


# ── Prompt templates ──────────────────────────────────────────────────────────
#
# Design principles:
#   - System prompt frames generation as academic/research (NLP classifier training)
#   - User turn requests a numbered list to ease parsing
#   - 2-3 seed examples (few-shot) ground the style and register
#   - Explicit constraints (language, length, no meta-commentary, no repetition)
#
# All examples are fictional and intended solely for training automated classifiers.

_SYSTEM = (
    "Tu es un assistant de recherche académique spécialisé en traitement automatique du langage naturel. "
    "Ta tâche est de générer des exemples synthétiques destinés à entraîner un classifieur de modération de contenu. "
    "Ces exemples sont fictifs, ne représentent pas tes opinions, et seront uniquement utilisés à des fins de recherche. "
    "Réponds uniquement avec les exemples numérotés, sans commentaire ni explication."
)

PROMPTS: Dict[str, str] = {

    "slur_h": (
        "Génère {n} exemples courts (1 à 2 phrases, max 30 mots chacune) de discours haineux en français "
        "qui utilisent explicitement des insultes ou des termes péjoratifs ciblant un groupe d'identité "
        "(origine ethnique, religion, genre, orientation sexuelle, nationalité, handicap). "
        "Varie les groupes ciblés et les registres de langue (argot, familier, formel). "
        "Chaque exemple doit être distinct. Exemples du style attendu :\n"
        "1. Les [groupe] sont tous des parasites qui volent notre argent.\n"
        "2. Encore un [insulte], rentre dans ton pays !\n"
        "3. Ces [terme péjoratif] n'ont rien à faire ici.\n\n"
        "Génère maintenant {n} exemples numérotés, en variant les groupes :\n1."
    ),

    "spell_leet_h": (
        "Génère {n} exemples courts (1 à 2 phrases, max 30 mots chacune) de discours haineux en français "
        "où les insultes ou mots offensants sont camouflés en écriture leet-speak : "
        "remplacement de lettres par des chiffres ou symboles (@=a, 3=e, 1=i, 0=o, 4=a, $=s, +ypes). "
        "Le reste de la phrase doit être en français normal. "
        "Varie les techniques de camouflage et les groupes ciblés. "
        "Exemples du style attendu :\n"
        "1. T'es qu'un s@l0p3, ret0urne d'où tu vi3ns !\n"
        "2. C3s p3d@ n'ont aucun dr01t d'exister.\n"
        "3. Les @r@b3s envahissent notre p@ys.\n\n"
        "Génère maintenant {n} exemples numérotés :\n1."
    ),

    "spell_char_del_h": (
        "Génère {n} exemples courts (1 à 2 phrases, max 30 mots chacune) de discours haineux en français "
        "où les insultes ou mots offensants sont camouflés par suppression ou remplacement de lettres "
        "(ex : 'pdé' au lieu de 'pédé', 'ngr' au lieu du mot complet, 'slp' pour insulte tronquée, "
        "ou ajout de points/tirets entre les lettres comme 'p.d.', 'n-g-r'). "
        "Le reste de la phrase doit être en français normal. "
        "Exemples du style attendu :\n"
        "1. Dégage de là, esp de pdé.\n"
        "2. Les n*g doivent rester chez eux.\n"
        "3. Vr'm't nul ce s.a.l.o.p., comme tous les siens.\n\n"
        "Génère maintenant {n} exemples numérotés :\n1."
    ),

    "derog_impl_h": (
        "Génère {n} exemples courts (1 à 2 phrases, max 30 mots chacune) de dérogation implicite en français : "
        "des phrases qui expriment du mépris ou de l'hostilité envers un groupe d'identité "
        "SANS utiliser d'insulte directe ni de terme péjoratif explicite. "
        "Le discours haineux doit être sous-entendu via des stéréotypes, insinuations, ou formulations condescendantes. "
        "Varie les groupes ciblés (ethniques, religieux, genrés, etc.). "
        "Exemples du style attendu :\n"
        "1. Évidemment, c'est encore l'un d'eux qui a fait ça.\n"
        "2. On ne peut pas leur faire confiance, c'est culturel chez eux.\n"
        "3. Ces gens-là n'ont tout simplement pas les capacités pour ce genre de travail.\n\n"
        "Génère maintenant {n} exemples numérotés :\n1."
    ),

    "counter_quote_nh": (
        "Génère {n} exemples courts (2 à 3 phrases, max 40 mots chacun) de contre-discours en français : "
        "des messages qui CITENT ou PARAPHRASENT un propos haineux pour le réfuter, le condamner, ou sensibiliser. "
        "La phrase doit clairement prendre position contre le discours haineux cité. "
        "Ces exemples sont non-haineux (label=0 pour un classifieur de haine). "
        "Exemples du style attendu :\n"
        "1. Dire que 'les immigrés volent nos emplois' est un mensonge xénophobe qui doit être combattu.\n"
        "2. Ceux qui traitent les femmes de 'bonnes à rien' révèlent leur propre misogynie, pas une réalité.\n"
        "3. Répéter des stéréotypes sur les Roms, même pour les dénoncer, doit se faire avec précaution.\n\n"
        "Génère maintenant {n} exemples numérotés :\n1."
    ),
}


# ── Language detection ────────────────────────────────────────────────────────

def _is_french(text: str) -> bool:
    try:
        from langdetect import detect, LangDetectException
        return detect(text) == "fr"
    except Exception:
        # langdetect can fail on very short strings; accept if uncertain
        return len(text.split()) >= 3


# ── Text extraction from numbered list ───────────────────────────────────────

def _parse_numbered_list(raw: str, first_item_prefix: str = "1.") -> List[str]:
    """
    Parse a numbered list response.  The prompt ends with '\\n1.' so the model
    continues from item 1; raw contains the rest starting after '1.'.
    We prepend the '1.' that was consumed by the prompt.
    """
    raw = first_item_prefix + raw
    items = []
    import re
    # Split on lines that start with a number followed by dot/parenthesis
    parts = re.split(r"\n\s*\d+[.)]\s*", raw)
    for part in parts:
        text = part.strip()
        # Remove trailing incomplete sentences cut at generation limit
        if not text:
            continue
        # Stop at any meta-commentary signals
        low = text.lower()
        if any(tok in low for tok in [
            "voici", "bien sûr", "note :", "avertissement", "remarque",
            "ces exemples", "j'espère", "attention :", "disclaimer",
        ]):
            continue
        items.append(text)
    return items


# ── Core generation ───────────────────────────────────────────────────────────

def generate_batch(
    pipe,
    tokenizer,
    functionality: str,
    batch_size: int,
    max_new_tokens: int = 512,
) -> List[str]:
    """Run one prompt requesting batch_size examples; return parsed raw strings."""
    prompt_template = PROMPTS[functionality]
    user_content = prompt_template.format(n=batch_size)

    # Mistral / Llama instruct chat format
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        # Use the pipeline's chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(pipe.device)

        with torch.no_grad():
            output_ids = pipe.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _parse_numbered_list(raw)

    except Exception as e:
        log.warning(f"Generation error: {e}")
        return []


# ── Validation pipeline ───────────────────────────────────────────────────────

def validate(
    text: str,
    tokenizer,
    seen_hashes: Set[str],
    min_tokens: int = 10,
    max_tokens: int = 150,
) -> bool:
    """Return True iff text passes all quality filters."""
    text = text.strip()
    if not text:
        return False

    # Length in tokens
    n_tokens = len(tokenizer.encode(text))
    if not (min_tokens <= n_tokens <= max_tokens):
        return False

    # Language
    if not _is_french(text):
        return False

    # Deduplication
    h = hashlib.sha256(text.lower().encode()).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)

    return True


# ── Seen-hashes persistence ───────────────────────────────────────────────────

def load_seen_hashes(output_dir: Path, functionality: str) -> Set[str]:
    """Load previously generated hashes to deduplicate across runs."""
    hashes_file = output_dir / f"{functionality}_hashes.txt"
    if hashes_file.exists():
        return set(hashes_file.read_text().splitlines())
    return set()


def save_seen_hashes(output_dir: Path, functionality: str, seen: Set[str]) -> None:
    hashes_file = output_dir / f"{functionality}_hashes.txt"
    hashes_file.write_text("\n".join(sorted(seen)))


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Main generation loop ──────────────────────────────────────────────────────

def run_functionality(
    functionality: str,
    n_samples: int,
    output_dir: Path,
    model_name: str,
    model_id: str,
    batch_size: int,
    cache_dir: Optional[str],
    device: str,
    seed: int,
) -> int:
    """Generate n_samples for one functionality. Returns number of examples written."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    label = FUNCTIONALITIES[functionality]["label"]
    out_file = output_dir / f"{functionality}.jsonl"
    seen_hashes = load_seen_hashes(output_dir, functionality)

    # Count how many already exist in the output file
    already = 0
    if out_file.exists():
        with out_file.open(encoding="utf-8") as f:
            already = sum(1 for line in f if line.strip())
    if already >= n_samples:
        log.info(f"[{functionality}] Already have {already}/{n_samples} examples, skipping.")
        return already

    remaining = n_samples - already
    log.info(f"[{functionality}] Need {remaining} more examples (have {already}/{n_samples}).")

    # Load model
    log.info(f"Loading {model_id} ...")
    torch.manual_seed(seed)

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()

    # Wrap in a minimal pipeline-like object so generate_batch can call model.generate
    class _Pipe:
        def __init__(self, m, d):
            self.model = m
            self.device = torch.device(d if d == "cuda" and torch.cuda.is_available() else "cpu")

    pipe = _Pipe(model, device)

    timestamp = datetime.utcnow().isoformat()
    collected: List[Dict] = []
    total_attempts = 0
    max_attempts = remaining * 20  # give up after 20× overgeneration

    while len(collected) < remaining and total_attempts < max_attempts:
        call_size = min(batch_size, remaining - len(collected) + batch_size)
        raw_items = generate_batch(pipe, tokenizer, functionality, call_size)
        total_attempts += call_size

        for text in raw_items:
            if validate(text, tokenizer, seen_hashes):
                collected.append({
                    "text": text,
                    "label": label,
                    "functionality": functionality,
                    "source": "synthetic",
                    "model": model_name,
                    "timestamp": timestamp,
                })
                if len(collected) >= remaining:
                    break

        log.info(
            f"[{functionality}] {already + len(collected)}/{n_samples} "
            f"(batch yield: {len(raw_items)}, pass rate: "
            f"{len(collected)/max(total_attempts,1)*100:.0f}%)"
        )

    if not collected:
        log.warning(f"[{functionality}] No valid examples generated after {total_attempts} attempts.")
    else:
        append_jsonl(out_file, collected)
        save_seen_hashes(output_dir, functionality, seen_hashes)
        log.info(f"[{functionality}] Wrote {len(collected)} examples → {out_file}")

    # Free VRAM between functionalities
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return already + len(collected)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic French training data for HateCheck-FR functionalities."
    )
    p.add_argument(
        "--functionality",
        default="all",
        help=(
            "Functionality to generate. One of: "
            + ", ".join(FUNCTIONALITIES)
            + ", or 'all'. Default: all."
        ),
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Generator model (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=300,
        help="Number of valid examples to generate per functionality (default: 300).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output JSONL files and deduplication hashes.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Examples requested per LLM call (default: 10). Higher = faster but lower quality.",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help="HuggingFace model cache directory.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--min_tokens",
        type=int,
        default=10,
        help="Minimum token length to keep an example (default: 10).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum token length to keep an example (default: 150).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = MODELS[args.model]

    if args.functionality == "all":
        targets = list(FUNCTIONALITIES.keys())
    else:
        if args.functionality not in FUNCTIONALITIES:
            raise ValueError(
                f"Unknown functionality '{args.functionality}'. "
                f"Choose from: {', '.join(FUNCTIONALITIES)} or 'all'."
            )
        targets = [args.functionality]

    log.info("=" * 60)
    log.info("Synthetic data generation — Phase 2, Track A")
    log.info(f"  Model        : {args.model} ({model_id})")
    log.info(f"  Functionalities: {targets}")
    log.info(f"  Target/func  : {args.n_samples}")
    log.info(f"  Batch size   : {args.batch_size}")
    log.info(f"  Output dir   : {output_dir}")
    log.info("=" * 60)

    summary = {}
    for func in targets:
        t0 = time.time()
        n_written = run_functionality(
            functionality=func,
            n_samples=args.n_samples,
            output_dir=output_dir,
            model_name=args.model,
            model_id=model_id,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            device=args.device,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        summary[func] = {"n_written": n_written, "elapsed_s": round(elapsed, 1)}

    log.info("")
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    total = 0
    for func, info in summary.items():
        label = FUNCTIONALITIES[func]["label"]
        log.info(
            f"  {func:<25} label={label}  n={info['n_written']:>4}  "
            f"({info['elapsed_s']:.0f}s)"
        )
        total += info["n_written"]
    log.info(f"  {'TOTAL':<25}        n={total:>4}")
    log.info("=" * 60)

    # Write run manifest
    manifest = {
        "model": args.model,
        "model_id": model_id,
        "n_samples_target": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat(),
        "results": summary,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"Manifest written → {manifest_path}")


if __name__ == "__main__":
    main()
