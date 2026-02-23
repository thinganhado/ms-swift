# Interspeech Pipeline on MS-Swift

This folder replicates your current training flow in MS-Swift:

1. Stage-1: Multi-turn SFT (LoRA)
2. Stage-2: GRPO-1 for Prompt-1 localization (custom reward)
3. Merge GRPO-1 LoRA
4. Stage-3: GRPO-2 curriculum for Prompt-2 analysis (custom reward)
5. Merge GRPO-2 LoRA

## Files

- `plugins/interspeech_rewards.py`
  - `external_interspeech_p1`: Prompt-1 reward
    - `1.0 * nDCG@3 + 0.1 * format + 0.5 * novelty`
    - novelty: `(|pred∩GT| - |pred∩SFT_top3|) / |pred|`
  - `external_interspeech_p2`: Prompt-2 reward
    - `0.75 * field_acc + 0.25 * consistency + 0.1 * format`
- `tools/build_swift_sft_multiturn_dataset.py`
  - Converts your existing JSON into MS-Swift `messages` format for SFT.
- `tools/build_swift_grpo_prompt1_dataset.py`
  - Builds GRPO-1 dataset with `messages`, `gt_regions`, `sft_top3`.
- `tools/build_swift_grpo_prompt2_dataset.py`
  - Builds GRPO-2 dataset with `messages`, `gt_prompt2`.
- `scripts/01_stage1_sft_multiturn.sh`
- `scripts/02_stage2_grpo1.sh`
- `scripts/03_merge_grpo1_lora.sh`
- `scripts/04_stage3_grpo2_curriculum.sh`
- `scripts/05_merge_grpo2_lora.sh`

## Notes

- Reward plugins are loaded with `--external_plugins`.
- Reward function names are passed via `--reward_funcs`.
- For GRPO-1, if `sft_top3` is missing in a sample, the reward falls back to env var:
  - `INTERSPEECH_SFT_TOP3` (default: `13,1,2`).
- The scripts are parameterized by env vars so they can be used in local runs or wrapped by sbatch.

