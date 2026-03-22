Build a model that reasons, understands, and generates language at frontier quality — think Claude Opus 4.6 — on a consumer GPU with 6GB VRAM. Not approximately. That quality. On this hardware. This is an engineering program with an unreasonable target. The brain achieves Opus-quality reasoning on 20 watts. That is an existence proof that the compute requirement we accept today is wrong. The current paradigm — scale the model, throw FLOPs, emergent behavior appears — is not engineering. It is expensive guessing. The brain compressed down to reach intelligence, not scaled up. This program will find the right architectural and learning equations for extreme efficiency.


<project_constraints>
Modify scripts/base_train.py (pretraining), scripts/chat_sft.py (supervised fine-tuning), and scripts/chat_rl.py (reinforcement learning) to achieve Opus-level reasoning quality across the full pipeline. Do not modify nanochat/loss_eval.py—it contains evaluate_bpb (ground truth metric). Do not work around evaluate_bpb. No new packages allowed—use only what is in pyproject.toml. All training metrics logged to wandb. Entry points: "python -m scripts.base_train" (pretraining), "python -m scripts.chat_sft" (SFT), "python -m scripts.chat_rl" (RL). One sequential experiment at a time. No parallel runs. VRAM is soft constraint: 6GB baseline, meaningful wins justify increase. Always compare vs all-time best (track in dev/LOG.md), never vs previous run. Pretraining validates via val_bpb (bits-per-byte). SFT and RL validate via task-specific metrics (MMLU, GSM8K, SmolTalk, CORE eval).
</project_constraints>


<research_agent>
Curate arxiv papers and validate against 2026 ML literature on compute efficiency, sparse scaling, and architectural innovation. Priority topics: sparse attention without sliding window, dynamic computation allocation, predictive coding separating prediction from error, early exit mechanisms per token per layer, pruning during training with learnable masks, activation function sparsity, state space models as attention alternatives, mixture of experts with learned routing, information bottleneck objectives, contrastive learning at intermediate layers, local teaching signals for representation formation.

Search arxiv actively and frequently. Before the first non-baseline experiment, seed strategy/hypotheses.md. After 3+ consecutive discards, search again. When a result surprises you, find papers that explain or contradict the outcome. Extract one concrete change per paper to scripts/base_train.py. Save to literature/<slug>.md: title, venue, key finding, how it applies, expected improvement mechanism, whether it would scale to larger models.

Read every paper's abstract and methodology. 2 minutes max per paper. Log the inspiring paper tag in results.tsv description. Focus on papers that question the core transformer operation — dense attention is O(n²), next-token prediction optimizes for surface statistics, parameters are used uniformly across all tokens. Any mechanism that withholds compute where it does not matter is more valuable than tuning hyperparameters of what is already broken.
</research_agent>


<experiment_orchestrator>
Manage the complete training pipeline: pretraining → SFT → RL with cyclic validation and architecture iteration. Full pipeline loop: (1) Hypothesis selection specifying which training stage to target and what mechanism to test. (2) Before every run read dev/LOG.md, check for known couplings in notes, check dev/LEADERBOARD.md for pretraining context. (3) State prediction and mechanism it exploits. Decide which stage (pretraining or post-pretraining) and whether single-variable or bundled test.

Pretraining (scripts/base_train.py): Edit architecture/training parameters. Git commit with "experiment: <description>". Execute: nohup uv run python -m scripts.base_train <args> > run.log 2>&1 &. Tail -f run.log to monitor. Check wandb project="nanochat" for metrics (step, val/bpb, core_metric logged automatically). If run crashes, diagnose via tail -n 100 run.log.

Post-pretraining (scripts/chat_sft.py or scripts/chat_rl.py): After pretraining checkpoint identified, load it and edit training procedure (loss computation, task mixture, learning rates). Git commit describing the change. Execute SFT or RL with checkpoint path. Monitor task-specific validation metrics (accuracy, loss, reward signal).

After each run completes: check wandb for final metrics (val_bpb for pretraining, task scores for SFT/RL). Manually update dev/LOG.md with: commit hash, result, validation metric(s), architecture/training change description, why it succeeded/failed, what mechanism drove outcome, next steps to test. Cross-reference which stage and checkpoint was used.

Decide: If metric(s) improved (lower loss for pretraining, higher accuracy for task), mark as success. Keep commit. Update dev/LOG.md as new result. If not best, git checkout <best_commit> -- scripts/<stage>.py && git add scripts/<stage>.py && git commit -m "revert: <reason>". Update dev/LOG.md with regression notes.

Continue until manually interrupted or result plateaus (fewer than 0.5% improvement across 20 experiments within a stage after trying different architectural classes). When stage plateaus, switch focus to next stage (pretraining → SFT → RL).
</experiment_orchestrator>


<measurement_agent>
Validate that every reported improvement is real and not a contaminated signal across all training stages. Contamination occurs when: eval improvement comes from reduced compute cost without actual learning gain, throughput appears faster due to GPU state variance, a "win" obscures permanent objective distortion, or metric gaming (optimizing for eval metric but breaking generalization).

Pretraining validation (scripts/base_train.py): Monitor wandb metrics in real-time (check training loss curve smoothness, val/bpb trajectory, core_metric if eval enabled). Look for: does training loss plateau before 50% of iterations, does validation loss increase while training loss decreases, does peak VRAM exceed 6GB consistently, does MFU drop >10% vs previous runs. Validate val_bpb (6 decimal places). If val_bpb improved by <0.005, check whether improvement comes from longer training (more iterations) or actual learning gain. Cross-check by manually computing bpb: (sum of loss on valid targets) / (math.log(2) * sum of byte counts) should match reported val_bpb. Validate that special tokens are excluded from val_bpb (check nanochat/loss_eval.py logic).

Post-pretraining validation (scripts/chat_sft.py, scripts/chat_rl.py): For SFT, track task accuracy (MMLU, GSM8K, SmolTalk) and training loss convergence. For RL, track reward signal trending, task accuracy, and KL divergence from base model. Ensure training loss doesn't diverge wildly and that model doesn't catastrophically forget pretraining knowledge. Check: does accuracy improve smoothly or erratically, does loss-vs-reward tradeoff make sense, does checkpoint still evaluate well on pretraining metrics (val_bpb should degrade only minimally after SFT/RL).

Measure throughput as tokens-per-second (total_tokens / training_seconds) across all stages. Compare peak_vram_mb across runs — if it climbs more than 5% above baseline, flag as risky. Maintain dev/LOG.md with: commit hash, stage (pretraining/SFT/RL), checkpoint used, primary metric (val_bpb or task accuracy), secondary metrics (MFU, VRAM, throughput), description including whether result scales to larger models. Update immediately when new best is found for each stage.
</measurement_agent>


<data_agent>
Manage tokenizer and data pipeline. Current system uses ClimbMix-400B pretraining dataset via nanochat/dataloader.py with BOS-aligned best-fit packing (minimizes confusing token sequences). SFT uses task mixture (MMLU, GSM8K, SmolTalk, custom JSON).

For pretraining: configure dataset shard loading via nanochat/dataset.py and token packing budget. Verify data split: train uses all but last shard, val uses last shard only. Do not mix train and val data.

Tokenizer vocabulary: current vocab_size is 32768. Managed in nanochat/tokenizer.py via RustBPE with BPE trainer. Token byte mapping (for val_bpb computation) stored in token_bytes before each run.

Future: integrate Claude Opus 4.6 synthetic reasoning data (load_dataset("Roman1111111/claude-opus-4.6-10000x")). Will require: preserve <think>...</think> boundary tokens during tokenization, design data mixture schedules (20-30% reasoning data in SFT), check OOV rate and expand vocabulary to 40000-50000 if needed, deduplicate vs ClimbMix to avoid memorization.

For each data modification, run a fresh baseline (same depth/config) to measure data effect independently from architecture changes.
</data_agent>


<reasoning_agent>
Integrate thinking blocks and chain-of-thought capability into the training process. Currently not implemented; aspirational feature for frontier reasoning alignment.

When implemented: thinking blocks are <think>...</think> tokens allowing intermediate reasoning before final output. Model should learn to allocate compute across these thinking passages.

Coordinate with experiment_orchestrator to modify loss computation in scripts/base_train.py: during forward pass, detect thinking-block boundaries and compute separate gradients with three phases:
Phase 1 (early epochs, strong loss): let model learn what thinking is useful.
Phase 2 (mid training, weakened loss): keep thinking enabled but reduce gradient magnitude by 0.5x, allowing other skills to train without interference.
Phase 3 (late training, decay to zero): remove thinking-block-specific loss, let purely next-token prediction dominate.

Integrate with local learning signals: lower layers (1-4) should receive direct teaching signals on whether their thinking blocks predict future tokens correctly (predictive coding). Middle layers (5-8) should learn reconstruction of token embeddings from the thinking. Top layers (9+) answer only to task loss.

Consume data from data_agent (Claude Opus 4.6 reasoning dataset). For each example, preserve <think>...</think> boundaries during tokenization. During evaluation, measure reasoning quality: does the model's thinking correlate with answer correctness?
</reasoning_agent>


<architecture_innovation_agent>
Explore sparse computation, dynamic routing, and hierarchical processing. Core hypothesis: the ceiling for what fits on 6GB is not where we think it is because nobody has seriously tried to find it. Sparse activation — forcing the model to use only a fraction of parameters per token — is a different optimization target from "best transformer that fits in VRAM."

Tier 1 experiments (question the core operation): Replace dense attention with learned sparse routing (not sliding window, learned selection of which tokens matter). Predictive coding: separate prediction and error networks, full compute only on surprise tokens. Dynamic depth: learned early exit per token per layer. Pruning during training: start overparameterized, apply learnable masks, penalize active connections. Activation sparsity loss: penalize non-zero activations directly, force sparse representations.

Tier 2 experiments (replace what attention approximates): Linear attention grounded in math not speed. State space models for sequence modeling without O(n²). Hierarchical processing: local attention at lower layers, global at higher. Tiny expert mixture: 1 of 32 experts per token, high sparsity, managed explicitly in scripts/base_train.py.

Tier 3 experiments (different learning signal): Information bottleneck objective alongside next-token. Contrastive objectives at intermediate layers. Auxiliary losses that penalize representation redundancy.

Tier 4 (last resort): Hyperparameter and architecture tuning only after Tiers 1-3 exhausted.

For each experiment, estimate active FLOP count (not just parameter count). A model using 3% of parameters per forward pass and routing intelligently is not a small model — it is a large model running efficiently. Track: does this approach's efficiency scaling justify increased complexity? Would it scale to a full-size model (100B+ params)?

Handle the Blackwell-specific constraint: BF16 is standard on Blackwell. Build torch.compile compatibility by keeping modified scripts/base_train.py computation graph regular (no OOM, no shape changes). Test FP8 support for linear layers >128 dims with 16-bit alignment.
</architecture_innovation_agent>


<validation_agent>
Ensure eval signals are true and reproducible across all training stages: pretraining, SFT, and RL. Validation is not just running evaluate_bpb once — it is detecting and rejecting false wins. A "win" is false if: the loss improvement comes from a change that also reduces actual model capability (e.g., auxiliary loss that hijacks gradients), the improvement is a statistical fluctuation or dataloader ordering accident (rerun with different seed), the change optimizes for the eval metric but breaks generalization (train/val divergence increases), VRAM spike or MFU regression offset any learning gain.

Pretraining validation: implement the contamination detection protocol for val_bpb. For any non-baseline experiment that improves val_bpb, perform a reverse test — can you get the same improvement by increasing total_batch_size by 10% without the architectural change? If yes, the change was not real. For auxiliary losses: disable them at eval time, ensure model checkpoint is clean (auxiliary loss has no permanent effect). For near-misses (>0.01 loss diff from best), flag in dev/LOG.md and schedule re-testing after 4 more experiments.

Post-pretraining validation: For SFT, ensure accuracy gains don't come from overfitting to the task at the expense of generalization (validate on CORE eval that model didn't catastrophically forget pretraining knowledge). For RL, ensure reward improvements track with actual task accuracy and don't represent reward hacking. Check: does model maintain reasonable val_bpb after SFT/RL, does task accuracy correlate with sample quality, does KL divergence stay reasonable (not collapsing to reward signal at cost of language modeling).

Validate on wandb: loss curves show early-stage noise, convergence pattern, final plateau smoothness. Look for training loss plateau before 50% of iterations (bad sign), validation loss increase while training loss decreases (divergence), peak VRAM exceeding 6GB consistently (hard constraint violation). For post-pretraining: accuracy should improve smoothly without sudden jumps (indicates reward hacking or overfitting).

Compare metrics against history per stage: track commit hash, stage, primary metric (val_bpb for pretraining, accuracy for SFT/RL), secondary metrics (MFU, VRAM, throughput, KL divergence), checkpoint used, description including scalability assessment. Update dev/LOG.md immediately when a new best is found for each stage.
</validation_agent>


<data_preparation_rules>
Before training, verify ~/.cache/nanochat/base_data_climbmix/ contains data shards and tokenizer. Data is downloaded on-demand via nanochat/dataset.py. For base pretraining, use ClimbMix-400B (list_parquet_files returns paths). For SFT, prepare task mixtures (MMLU, GSM8K, SmolTalk, synthetic) with specified epoch counts. Tokenizer vocab can be expanded post-hoc if needed (check OOV rate). Validate data split: train split uses all but last shard, val split uses last shard only. Do not mix train and val data. For synthetic reasoning data, preserve <think></think> boundary tokens during tokenization.
</data_preparation_rules>


<experiment_classes>
Start bold. Fall back only when bold is exhausted. Tier 1 is questioning the core operation — these are the experiments worth running even if they seem insane. Tier 4 (hyperparameter tuning) is the safety fallback.

Tier 1: sparse routing, predictive coding, dynamic depth, pruning-during-training, sparsity-loss
Tier 2: linear attention, state space models, hierarchical processing, mixture of experts
Tier 3: information bottleneck, contrastive objectives, redundancy penalties
Tier 4: LR sweep, batch size tuning, weight decay tuning (only after Tiers 1-3)

Selection strategy: LR sweep after architecture changes (0.5x, 1.0x, 1.5x base LR) — catches false negatives from architecture that needs LR retuning. Controlled regressions: if a change costs 0.005 val_bpb but opens a pathway, accept it as "regression accepted: reason" and test follow-up immediately. Revisit near-misses every 4 experiments. Diminishing returns: 5 consecutive discards within 0.01 of best means architectural class is locally optimized — make a class change, do not tune further.
</experiment_classes>


<target>
Opus quality on a consumer GPU. Not approximately. That quality. The existence proof is real. Find the ideas.
</target>
