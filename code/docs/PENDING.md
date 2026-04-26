
  # Group 1 — scoring scripts + sbatch
  scp code/phase4_two_tier/score_two_tier.py \
      alan:~/code/phase4_two_tier/score_two_tier.py

  scp code/phase4_two_tier/simulate_thresholds.py \
      alan:~/code/phase4_two_tier/simulate_thresholds.py

  scp code/phase4_two_tier/slurm/score_two_tier_pretrained.sbatch \
      alan:~/code/phase4_two_tier/slurm/score_two_tier_pretrained.sbatch

  scp code/phase4_two_tier/slurm/score_two_tier_finetuned.sbatch \
      alan:~/code/phase4_two_tier/slurm/score_two_tier_finetuned.sbatch

  # Group 2 — modified training script + new sbatch
  scp code/phase4_two_tier/finetune_detoxify_tier1.py \
      alan:~/code/phase4_two_tier/finetune_detoxify_tier1.py

  scp code/phase4_two_tier/slurm/finetune_tier1_v2.sbatch \
      alan:~/code/phase4_two_tier/slurm/finetune_tier1_v2.sbatch

  Then submit both groups simultaneously:
  sbatch ~/code/phase4_two_tier/slurm/score_two_tier_pretrained.sbatch
  sbatch ~/code/phase4_two_tier/slurm/score_two_tier_finetuned.sbatch
  sbatch ~/code/phase4_two_tier/slurm/finetune_tier1_v2.sbatch