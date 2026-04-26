#!/usr/bin/env python3
"""
generate_classifier_data.py — Synthetic data for Shareish content-type classifier
                               (Phase 3, Track B supplement)

Generates synthetic French text examples for a binary classifier:
    label 1 = solidarity_exchange  (free offers/requests, mutual aid, community)
    label 0 = commercial_listing   (for-profit sales, price-tagged, transactional)

Why synthetic data is needed:
    - donnons.org scraped 3,238 solidarity examples (well covered)
    - 2ememain.be only yielded 51 commercial examples (critically underrepresented)
    - Synthetic data bridges the gap until scrapers are fixed

Design principles:
    1. Category diversity — the same object types appear in both classes so the
       classifier learns *framing intent*, not object vocabulary.
    2. Hard examples — each class includes boundary cases that are easy to confuse
       (e.g. a charity sale at token price for solidarity; an emphatic free offer
       with commercial-sounding detail for commercial).
    3. Belgian French register — prompts target the Shareish deployment context.

Output JSONL fields: text, label, functionality, category, source, model, timestamp

Usage:
    python code/phase3_data/generate_classifier_data.py \\
        --output_dir ~/code/data/synthetic_classifier \\
        --n_samples 300 \\
        --model mistral_7b_instruct

    # One class only:
    python code/phase3_data/generate_classifier_data.py \\
        --output_dir ~/code/data/synthetic_classifier \\
        --classes commercial_listing \\
        --n_samples 500
"""

import gc
import json
import time
import random
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

MODELS = {
    "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral_7b_instruct_v2": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

DEFAULT_MODEL = "mistral_7b_instruct"


# ── Class config ──────────────────────────────────────────────────────────────

CLASSES = {
    "solidarity_exchange": {"label": 1},
    "commercial_listing":  {"label": 0},
}


# ── Object categories ─────────────────────────────────────────────────────────
# The same list is used for both classes so the model sees identical object
# vocabulary and learns framing intent rather than domain-specific words.

CATEGORIES = [
    "meubles (canapé, lit, armoire, table, chaise)",
    "électroménager (lave-linge, réfrigérateur, micro-ondes, aspirateur)",
    "vêtements et chaussures (manteau, jeans, chaussures de sport, robes)",
    "puériculture (poussette, siège auto, lit de bébé, jouets)",
    "informatique et téléphonie (ordinateur portable, smartphone, tablette)",
    "livres et multimédia (romans, BD, jeux vidéo, DVD)",
    "sports et loisirs (vélo, tapis de yoga, raquettes, ski)",
    "jardin et bricolage (outils, plantes, tondeuse, établi)",
    "arts de la table et décoration (vaisselle, cadres, lampes, tapis)",
    "services et aide (déménagement, jardinage, cours, transport)",
]


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = (
    "Tu es un assistant de recherche académique spécialisé en traitement automatique "
    "du langage naturel. Ta tâche est de générer des exemples synthétiques destinés à "
    "entraîner un classifieur de type de contenu pour une plateforme d'entraide belge. "
    "Ces exemples sont fictifs et utilisés uniquement à des fins de recherche. "
    "Réponds uniquement avec les exemples numérotés, sans commentaire ni explication."
)


# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPTS = {

    "solidarity_exchange": (
        "Génère {n} exemples courts (2 à 4 phrases, max 60 mots chacun) de messages "
        "d'échange solidaire en français belge, du type qu'on trouve sur Shareish ou donnons.org. "
        "Ces messages proposent ou demandent des objets/services GRATUITEMENT, dans un esprit "
        "communautaire et d'entraide. Varie les catégories d'objets parmi : {category}. "
        "Varie aussi les formulations (donner, chercher, offrir, proposer, récupérer). "
        "Inclus 1 ou 2 exemples 'limite' : une vente symbolique à prix libre ou 1€ "
        "mais avec une framing solidaire explicite.\n\n"
        "Exemples du style attendu :\n"
        "1. Je donne un canapé 2 places en bon état, tissu gris. À venir récupérer à "
        "Ixelles avant samedi. Idéal pour quelqu'un qui en a besoin !\n"
        "2. Cherche quelqu'un pour m'aider à déménager samedi matin, pas de budget mais "
        "je cuisine pour tout le monde en échange.\n"
        "3. Offre cours de français gratuits pour personnes réfugiées, 1h/semaine à Liège.\n\n"
        "Génère maintenant {n} exemples numérotés en variant les catégories :\n1."
    ),

    "commercial_listing": (
        "Génère {n} exemples courts (2 à 4 phrases, max 60 mots chacun) de petites annonces "
        "de vente en français belge, du type qu'on trouve sur 2ememain.be ou leboncoin.fr. "
        "Ces annonces vendent des objets d'occasion À UN PRIX, dans un cadre transactionnel. "
        "Varie les catégories d'objets parmi : {category}. "
        "Inclus des détails commerciaux : prix, état, livraison/enlèvement, négociation. "
        "Inclus 1 ou 2 exemples 'limite' : un objet donné presque gratuitement (5–10€) "
        "mais avec framing clairement commercial (prix fixe, pas de négociation).\n\n"
        "Exemples du style attendu :\n"
        "1. Je vends mon lave-linge Samsung 7kg, très bon état, 2 ans d'utilisation. "
        "Prix : 150€ ferme. Enlèvement uniquement à Liège, pas d'envoi.\n"
        "2. À vendre : lot de vêtements femme taille 38, marques variées. 40€ le lot, "
        "pas de séparation. Disponible en semaine.\n"
        "3. Vends vélo de ville 26 pouces, révisé récemment, antivol inclus. 80€. "
        "Rdv possible à Bruxelles ou Namur.\n\n"
        "Génère maintenant {n} exemples numérotés en variant les catégories :\n1."
    ),
}


# ── Language check ────────────────────────────────────────────────────────────

def _is_french(text: str) -> bool:
    try:
        from langdetect import detect
        return detect(text) == "fr"
    except Exception:
        return len(text.split()) >= 4


# ── List parser ───────────────────────────────────────────────────────────────

def _parse_numbered_list(raw: str) -> List[str]:
    """Parse a numbered list response, stripping item numbers and meta-commentary."""
    import re
    # Prepend the "1." consumed by the prompt ending
    raw = "1." + raw
    parts = re.split(r"\n\s*\d+[.)]\s*", raw)
    items = []
    for part in parts:
        text = part.strip()
        # Strip any residual leading number artifact (e.g. "1. " or "1.1. ")
        text = re.sub(r"^\d+\.\d*\s*", "", text).strip()
        if not text:
            continue
        low = text.lower()
        if any(tok in low for tok in [
            "voici", "bien sûr", "note :", "remarque :", "ces exemples",
            "j'espère", "attention :", "disclaimer", "avertissement",
        ]):
            continue
        items.append(text)
    return items


# ── Core generation ───────────────────────────────────────────────────────────

def generate_batch(
    model,
    tokenizer,
    device: torch.device,
    cls: str,
    batch_size: int,
    max_new_tokens: int = 600,
) -> List[str]:
    """Request batch_size examples for one class; return parsed strings."""
    category = random.choice(CATEGORIES)
    prompt_template = PROMPTS[cls]
    user_content = prompt_template.format(n=batch_size, category=category)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _parse_numbered_list(raw)

    except Exception as e:
        log.warning(f"Generation error ({cls}): {e}")
        return []


# ── Validation ────────────────────────────────────────────────────────────────

def validate(
    text: str,
    tokenizer,
    seen_hashes: Set[str],
    min_tokens: int = 10,
    max_tokens: int = 150,
) -> bool:
    text = text.strip()
    if not text:
        return False
    n_tokens = len(tokenizer.encode(text))
    if not (min_tokens <= n_tokens <= max_tokens):
        return False
    if not _is_french(text):
        return False
    h = hashlib.sha256(text.lower().encode()).hexdigest()
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True


# ── Checkpointing ─────────────────────────────────────────────────────────────

def load_seen_hashes(output_dir: Path, cls: str) -> Set[str]:
    f = output_dir / f"{cls}_hashes.txt"
    return set(f.read_text().splitlines()) if f.exists() else set()


def save_seen_hashes(output_dir: Path, cls: str, seen: Set[str]) -> None:
    (output_dir / f"{cls}_hashes.txt").write_text("\n".join(sorted(seen)))


def count_existing(output_dir: Path, cls: str) -> int:
    f = output_dir / f"{cls}.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def append_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Main generation loop ──────────────────────────────────────────────────────

def run_class(
    cls: str,
    n_samples: int,
    output_dir: Path,
    model_name: str,
    model_id: str,
    batch_size: int,
    cache_dir: Optional[str],
    device_str: str,
    seed: int,
) -> int:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    label = CLASSES[cls]["label"]
    out_file = output_dir / f"{cls}.jsonl"
    seen_hashes = load_seen_hashes(output_dir, cls)

    already = count_existing(output_dir, cls)
    if already >= n_samples:
        log.info(f"[{cls}] Already have {already}/{n_samples}. Skipping.")
        return already

    remaining = n_samples - already
    log.info(f"[{cls}] Need {remaining} more examples (have {already}/{n_samples}).")

    torch.manual_seed(seed)
    log.info(f"Loading {model_id} ...")

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        device_map="auto" if device_str == "cuda" else None,
    )
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    timestamp = datetime.utcnow().isoformat()
    collected: List[Dict] = []
    total_attempts = 0
    max_attempts = remaining * 20

    while len(collected) < remaining and total_attempts < max_attempts:
        raw_items = generate_batch(model, tokenizer, device, cls, batch_size)
        total_attempts += batch_size

        for text in raw_items:
            if validate(text, tokenizer, seen_hashes):
                collected.append({
                    "text": text,
                    "label": label,
                    "functionality": cls,
                    "source": "synthetic",
                    "model": model_name,
                    "timestamp": timestamp,
                })
                if len(collected) >= remaining:
                    break

        log.info(
            f"[{cls}] {already + len(collected)}/{n_samples} "
            f"(pass rate: {len(collected)/max(total_attempts,1)*100:.0f}%)"
        )

    if collected:
        append_jsonl(out_file, collected)
        save_seen_hashes(output_dir, cls, seen_hashes)
        log.info(f"[{cls}] Wrote {len(collected)} examples → {out_file}")
    else:
        log.warning(f"[{cls}] No valid examples after {total_attempts} attempts.")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return already + len(collected)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic data for the Shareish content-type classifier."
    )
    p.add_argument(
        "--classes", default="all",
        help="Comma-separated classes or 'all'. "
             "Options: solidarity_exchange, commercial_listing. Default: all.",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
        help=f"Generator model (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--n_samples", type=int, default=300,
        help="Valid examples to generate per class (default: 300).",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Examples requested per LLM call (default: 8).",
    )
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = MODELS[args.model]

    if args.classes == "all":
        targets = list(CLASSES.keys())
    else:
        targets = [c.strip() for c in args.classes.split(",")]
        unknown = [c for c in targets if c not in CLASSES]
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}. Choose from: {list(CLASSES)}")

    log.info("=" * 60)
    log.info("Synthetic classifier data — Phase 3, Track B")
    log.info(f"  Model   : {args.model}")
    log.info(f"  Classes : {targets}")
    log.info(f"  Target  : {args.n_samples}/class")
    log.info(f"  Output  : {output_dir}")
    log.info("=" * 60)

    summary = {}
    for cls in targets:
        t0 = time.time()
        n = run_class(
            cls=cls,
            n_samples=args.n_samples,
            output_dir=output_dir,
            model_name=args.model,
            model_id=model_id,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            device_str=args.device,
            seed=args.seed,
        )
        summary[cls] = {"label": CLASSES[cls]["label"], "n": n, "elapsed_s": round(time.time()-t0, 1)}

    log.info("")
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)
    for cls, info in summary.items():
        log.info(f"  {cls:<25} label={info['label']}  n={info['n']:>4}  ({info['elapsed_s']:.0f}s)")
    log.info("=" * 60)

    manifest = {
        "model": args.model,
        "model_id": model_id,
        "n_samples_target": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat(),
        "results": summary,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"Manifest → {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
