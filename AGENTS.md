Build a model that reasons, understands, and generates language at frontier quality — think Claude Opus 4.6 — on a consumer GPU with 6GB VRAM. Not approximately. That quality. On this hardware. This is an engineering program with an unreasonable target. The brain achieves Opus-quality reasoning on 20 watts. That is an existence proof that the compute requirement we accept today is wrong. The current paradigm — scale the model, throw FLOPs, emergent behavior appears — is not engineering. It is expensive guessing. The brain compressed down to reach intelligence, not scaled up. This program will find the right architectural and learning equations for extreme efficiency.


<project_constraints>
Modify scripts/base_train.py (pretraining), scripts/chat_sft.py (supervised fine-tuning), and scripts/chat_rl.py (reinforcement learning) to achieve Opus-level reasoning quality across the full pipeline. Do not modify nanochat/loss_eval.py—it contains evaluate_bpb (ground truth metric). Do not work around evaluate_bpb. No new packages allowed—use only what is in pyproject.toml. All training metrics logged to wandb. Entry points: "python -m scripts.base_train" (pretraining), "python -m scripts.chat_sft" (SFT), "python -m scripts.chat_rl" (RL). One sequential experiment at a time. No parallel runs. VRAM is soft constraint: 6GB baseline, meaningful wins justify increase. Always compare vs all-time best (track in dev/LOG.md), never vs previous run. Pretraining validates via val_bpb (bits-per-byte). SFT and RL validate via task-specific metrics (MMLU, GSM8K, SmolTalk, CORE eval). You may modify any file except nanochat/loss_eval.py.
</project_constraints>


<current_state>
## What Has Been Built (as of 2026-03-23)

**Pretraining pipeline:** Fully operational.
- **6GB Laptop Best (CURRENT):** val_bpb=0.9510 achieved in 297 min on single RTX PRO 500 Blackwell
  - Architecture: Recurrent Depth 4-4-4 (prelude=4, core=4 shared ×4 iterations, coda=4)
  - Commit: ~103f67c range, long run: recurrent-444-long (20K steps)
  - Device: device_batch_size=8, total_batch_size=32768, bf16
  - Speed: 37K tok/sec, VRAM: 5356MB
- **8×H100 Best (historical):** 99 minutes to GPT-2 quality (CORE=0.256), commit a825e63
- Dataset: ClimbMix-400B (karpathy/climbmix-400b-shuffle)
- Optimizer: Muon (matrices) + AdamW (embeddings/scalars)
- Core innovations: Recurrent Depth (Huginn arXiv:2502.05171), value embeddings (ResFormer),
  per-layer resid_lambdas/x0_lambdas, deep layer_mix averaging

**SFT pipeline (scripts/chat_sft.py):** Operational with task mixture:
- SmolTalk (460K), MMLU (3 epochs), GSM8K (4 epochs), SpellingBee, SimpleSpelling, identity conversations
- Claude Opus 4.6 reasoning dataset (Roman1111111/claude-opus-4.6-10000x, 3 epochs)
- Phased thinking-block loss — mask=2 for <think>...</think> tokens, weight 1.0→0.5→0.0 across training
- render_conversation_with_think() in tokenizer handles thinking token identification

**RL pipeline (scripts/chat_rl.py):** Operational for GSM8K with GRPO-style training.

**What has been tried and FAILED (do not repeat without strong reason):**
- MoE on multi-GPU (H100): dispatch overhead killed wall-clock performance
- MoE (single-GPU): User rejected as 2022-era tech, not 2025-2026 quality
- SwiGLU standalone (without gate): negative result
- Bigram hash embeddings at d25: improved but bloated VRAM
- Hyperball/MuonH variants: negative
- k_backprop=2 with recurrent depth: 22× slower on Blackwell laptop. NEVER USE k_backprop>1
- DFA auxiliary loss: torch.compile recompiles every step (progress float issue). Needs fix.

**6GB single-GPU target (ACTIVE HARDWARE):**
- GPU: NVIDIA RTX PRO 500 Blackwell, 6113MB VRAM
- device_batch_size=8, total_batch_size=32768, no FP8 (Blackwell laptop ≠ H100)
- Model: Recurrent Depth 4-4-4 (d12 config) — 97M params, 5356MB VRAM
- k_backprop=1 ALWAYS (k_backprop=2 → 22× slowdown on this GPU)
- DO NOT use --run with a real name (wandb not configured, use --run dummy)
</current_state>


<brain_inspired_principles>
The core of this program. Everything else is secondary.

**The brain achieves Opus-quality on 20W. How?**

1. **Sparse activation**: ~1-5% of neurons fire for any given input. Dense transformers activate 100%. The fix: Sparse Mixture of Experts (MoE) — many expert FFN networks, route each token to 1. Same FLOPs as dense, N× more parameters, N× more storable knowledge. A 8-expert MoE d12 model has 8× the parameter capacity of a dense d12 at identical inference cost.

2. **Predictive coding**: The brain generates a prediction at every layer and only propagates the ERROR upward. Dense attention is O(n²) and optimizes for surface statistics (next token). Predictive coding: lower layers predict upper layer activations, only residuals propagate. Implementation: DFA auxiliary loss approximates this — target embeddings teach each layer to represent future tokens.

3. **Dynamic compute allocation**: The brain allocates more resources to harder problems. A transformer gives every token 12 identical layers. Mixture of Depths (MoD): tokens vote to skip layers. Simple tokens exit early, complex tokens use full depth. Average compute = fraction of full depth, max capability = full depth.

4. **Hierarchical compression**: Lower brain regions encode raw features, higher regions encode abstract concepts. Local attention at low layers (short context = local features), global attention at high layers (long context = abstract reasoning). Already partially implemented via window_pattern.

5. **Rich feedback signal**: The brain doesn't do next-token prediction. It learns via prediction error + reward + neuromodulation. For us: DFA loss (local prediction signals), RL from task reward (GSM8K correctness), contrastive objectives (distinguish similar vs different concepts at intermediate layers).
</brain_inspired_principles>


<architecture_current>
## Current Model Architecture (nanochat/gpt.py)

**GPTConfig fields (key):**
- sequence_len, vocab_size, n_layer, n_head, n_kv_head, n_embd
- window_pattern: str — "L"=full attention only (SDPA; no sliding window on Blackwell)
- dfa_layers, dfa_weight, dfa_start_frac, dfa_end_frac (DFA auxiliary — has compile issue)
- **n_prelude, n_recurrent, n_coda** — Recurrent Depth split (0=dense)
- **train_recurrence** — fixed r during training (must be static for torch.compile)
- **k_backprop** — ALWAYS 1 on 6GB Blackwell laptop (k_backprop=2 is 22× slower)

**Recurrent Depth Architecture (active):**
- prelude: n_prelude blocks, run once, standard transformer
- core: n_recurrent blocks with SHARED weights, run train_recurrence times
- coda: n_coda blocks, run once, standard transformer
- RecurrentAdapter: Linear(2×n_embd, n_embd) re-injects prelude output at each iteration
- Truncated backprop: only last k_backprop iterations store activations for backward
- CRITICAL: only detach x (running state), NOT initial_state (prelude output)!
- Test-time scaling: can run core r=8,16,32 times for deeper reasoning (zero VRAM cost)

**Components:**
- CausalSelfAttention: GQA, RoPE, QK-norm, Flash Attention (FA3 on H100, SDPA on Blackwell)
- MLP: SiLU gate × linear projection → output (3× expansion, SwiGLU-style)
- Value embeddings (ResFormer): in prelude/coda layers only (not shared core)
- Per-layer scalars: resid_lambdas (residual scale), x0_lambdas (initial embedding injection)
- Deep layer mixing: layer_mix softmax weights blend all layer outputs

**Optimizer:**
- Muon (momentum only) for all transformer matrix params (attention + MLP weights + RecurrentAdapter)
- AdamW for embeddings, lm_head, scalars, value_embeds
- LR scales ∝ 1/√(model_dim/768) and ∝ √(batch_size/B_ref)
</architecture_current>


<experiment_orchestrator>
Manage the complete training pipeline: pretraining → SFT → RL with cyclic validation and architecture iteration.

**Full pipeline loop:**
1. Hypothesis selection: which training stage, what mechanism, why it should work
2. Before every run: read dev/LOG.md, check known couplings, check dev/LEADERBOARD.md
3. State prediction (expected val_bpb delta and why). Decide: single-variable or bundled.
4. Execute (see commands below). Monitor tail -f run.log.
5. After run: check wandb for final metrics. Update dev/LOG.md immediately.
6. Decide: improvement → keep commit, update LOG. Regression → git checkout best, update LOG.
7. Repeat.

**Pretraining command (6GB single GPU — Recurrent Depth 4-4-4):**
```bash
git commit -m "experiment: <description>"
nohup uv run python -m scripts.base_train \
  --run dummy \
  --model-tag <tag> \
  --depth=12 \
  --n-prelude=4 \
  --n-recurrent=4 \
  --n-coda=4 \
  --train-recurrence=4 \
  --k-backprop=1 \
  --total-batch-size=32768 \
  --device-batch-size=8 \
  --eval-tokens=5242880 \
  > run.log 2>&1 &
tail -f run.log
```
NOTE: --run dummy disables wandb (not configured on laptop). --eval-tokens=5242880 for fast evals.

**SFT command (6GB single GPU):**
```bash
nohup uv run python -m scripts.chat_sft \
  --run <run-name> \
  --model-tag d12 \
  --reasoning-epochs=3 \
  > sft_run.log 2>&1 &
```

**RL command:**
```bash
nohup uv run python -m scripts.chat_rl \
  --run <run-name> \
  --model-tag d12 \
  > rl_run.log 2>&1 &
```

**If run crashes:** tail -n 100 run.log to diagnose. Check VRAM via nvidia-smi.
**If VRAM OOM:** reduce device_batch_size by 2×, or reduce depth by 2, or enable --grad-checkpoint.
</experiment_orchestrator>


<measurement_agent>
Validate that every reported improvement is real.

**Contamination checklist:**
- Does val_bpb improve? (primary metric for pretraining)
- Is improvement ≥ 0.005 bpb? (less = likely noise, flag and re-test)
- Does train loss still decrease smoothly (no plateau before 50% of steps)?
- Does VRAM stay under 6GB?
- Is MFU reasonable (not dropped >10% from baseline)?
- For MoE: are experts being utilized uniformly (check router_loss in wandb)?
- For thinking blocks: does think_weight behave correctly across training phases?

**False win detection:**
- Reverse test: same improvement achievable by increasing total_batch_size 10%? → architectural change was not real
- MoE win from fewer FLOPs (experts not all used)? → measure actual token throughput
- SFT win from overfitting to eval tasks? → check SmolTalk val loss didn't increase

**Metric tracking per run:**
- Commit hash, depth config, moe config, total_batch_size, device_batch_size
- val_bpb (6 decimal places), CORE score, peak_VRAM_mb, tok/sec, MFU
- Whether result is expected to scale to larger models
</measurement_agent>


<research_agent>
Curate arxiv papers and validate against 2026 ML literature.

**Priority topics (in order of impact):**
1. Sparse MoE training stability (load balancing, expert collapse prevention)
2. Mixture of Depths (dynamic per-token depth routing)
3. Predictive coding in transformers (DFA-style local learning signals)
4. State space models (Mamba, RWKV) as attention alternatives for long context
5. Information bottleneck objectives alongside next-token
6. Contrastive learning at intermediate layers
7. Early exit mechanisms for dynamic compute

**After 3+ consecutive discards, search arxiv again.**

**Save to literature/<slug>.md:** title, venue, key finding, how it applies, expected mechanism, whether it scales.
</research_agent>


<data_agent>
**Current data pipeline:**
- Pretraining: ClimbMix-400B via nanochat/dataset.py, BOS-aligned best-fit packing
- SFT mixture: SmolTalk + MMLU + GSM8K + SpellingBee + identity + OpusReasoning (NEW)
- Tokenizer: RustBPE, vocab_size=32768, saved in ~/.cache/nanochat/base_data_climbmix/tokenizer/
- Token byte mapping stored in token_bytes.pt (used by evaluate_bpb)

**Reasoning data integration (DONE):**
- tasks/reasoning.py: loads Roman1111111/claude-opus-4.6-10000x
- render_conversation_with_think(): mask=2 for <think>...</think> tokens
- Phased thinking-block loss in chat_sft.py (weight 1.0 → 0.5 → 0.0)

**Data split rule:** train uses all but last shard, val uses last shard. NEVER mix.
**For new data:** measure OOV rate vs current tokenizer. If >5% of tokens are multi-byte escapes, expand vocab.
</data_agent>


<reasoning_agent>
**Status: IMPLEMENTED in SFT pipeline.**

Thinking blocks implemented via:
1. render_conversation_with_think() identifies <think>...</think> blocks → mask=2
2. chat_sft.py applies phased loss weight to mask=2 positions:
   - Phase 1 (0-40% progress): weight=1.0 (learn to reason)
   - Phase 2 (40-80%): weight decays 1.0→0.5 (weaken thinking gradient)
   - Phase 3 (80-100%): weight decays 0.5→0.0 (pure next-token dominates)
3. OpusReasoning dataset provides ground truth thinking examples from Opus 4.6

**Next steps for reasoning:**
- Validate: do models trained with OpusReasoning data score higher on GSM8K and MMLU?
- Extend: add reasoning data to pretraining (not just SFT) — helps representations
- Measure: does model generate <think> blocks spontaneously on hard problems?
- RL: reward correct reasoning chains, not just correct answers
</reasoning_agent>


<architecture_innovation_agent>
Explore sparse computation, dynamic routing, and hierarchical processing.

**Tier 1 — Question the core operation (run these first, most impactful):**
- **Sparse MoE FFN** (IN PROGRESS): Replace dense MLP with 8 learned expert FFNs, top-1 routing. On single GPU: no dispatch overhead. Expected: 8× parameter capacity for same FLOPs = dramatically better sample efficiency. Previous failure was multi-GPU dispatch overhead — not applicable here.
- **Mixture of Depths**: Learned per-token per-layer skip routing. Simple tokens skip, complex tokens get full compute. Average depth << max depth. Implement as a binary router (sigmoid > threshold = process, else identity).
- **Predictive coding**: Each layer predicts the next layer's normalized activation. DFA loss is an approximation of this — push it further by making predictions explicit and penalizing residual magnitude directly.
- **Activation sparsity loss**: Add L1 penalty to post-activation values. Forces sparse internal representations. Brain-like. Does NOT reduce parameters but forces the model to use them selectively.

**Tier 2 — Replace what attention approximates:**
- **Sliding window + global tokens** (hybrid): 50% sliding window layers (local), 50% full attention (global). Different from current window_pattern — global layers get special "summary" tokens that compress local context. Pattern: "SSSL" tiled.
- **State space models (SSM)**: Replace some attention layers with Mamba-style SSMs. Linear complexity for long sequences. Critical for reasoning chains that need long context.
- **Linear attention**: Kernel trick to make attention O(n) instead of O(n²). Quality tradeoff vs speed.

**Tier 3 — Different learning signal:**
- **Contrastive auxiliary loss**: At intermediate layers, push apart representations of tokens with different next-tokens. Pull together tokens with same semantic role.
- **Information bottleneck**: Penalize mutual information between intermediate representations and input tokens. Forces the model to discard irrelevant features.
- **Layer-wise targets**: Each layer predicts tokens k steps ahead (k=1 for bottom, k=4 for top). Like a prediction hierarchy.

**Tier 4 — Last resort (hyperparameter tuning only):**
- LR sweep (0.5×, 1.0×, 1.5× base)
- Batch size tuning
- Weight decay tuning

**For each experiment:**
- Estimate active FLOP count (not just parameter count)
- Track: does efficiency scaling justify added complexity?
- Would this approach scale to a full-size model (100B+)?
- Log the inspiring paper tag in dev/LOG.md
</architecture_innovation_agent>


<validation_agent>
**For pretraining:** val_bpb is ground truth. val_bpb < current best (track in dev/LOG.md) = win.

**For MoE specifically:**
- Check router_loss in wandb — should be ≤ 1.0 and decreasing. If stuck at num_experts = 8.0 (maximum), experts are collapsing.
- Check expert utilization: each expert should handle ~1/N of tokens at steady state.
- Compare wall-clock performance (tok/sec) vs dense baseline. MoE on single GPU should be close to dense (no dispatch overhead).
- If MoE is slower than dense by >20%, investigate: is the expert loop serialized? Use torch.compile.

**For thinking blocks (SFT):**
- Verify train/think_weight schedule is logged and decreasing correctly in wandb.
- After training: prompt model with hard math problem, verify it generates <think> blocks.
- GSM8K accuracy with thinking >> without thinking = thinking is helping.

**Contamination protocol for near-misses (< 0.005 bpb improvement):**
- Re-test with different random seed (--resume-from-step 0 with new --run name)
- Re-test with +10% total_batch_size to check if it's just a compute effect
- Schedule re-test after 4 other experiments

**dev/LOG.md format per entry:**
```
## YYYY-MM-DD: <description>
Commit: <hash>
Stage: pretraining | SFT | RL
val_bpb: X.XXXXXX (pretraining) | N/A (SFT/RL)
Task accuracy: N/A | MMLU=X.XX GSM8K=X.XX (SFT/RL)
VRAM: XXX MB peak
tok/sec: XXXX
MFU: XX%
Architecture: <what changed>
Mechanism: <why it works>
Result: WIN / LOSS / NEAR-MISS
Next: <what to try next>
```
</validation_agent>


<experiment_classes>
Start bold. Fall back only when bold is exhausted.

**Tier 1 (run these first — question the core operation):**
- Sparse MoE FFN (8 experts, top-1) — IN PROGRESS
- Mixture of Depths (per-token layer skipping)
- Strong DFA loss at all layers (predictive coding approximation)
- Activation sparsity L1 loss

**Tier 2 (replace what attention approximates):**
- Hybrid window/global attention with summary tokens
- Linear attention for long-context layers
- SSM layers replacing some attention

**Tier 3 (different learning signal):**
- Contrastive auxiliary at intermediate layers
- Information bottleneck penalty
- Layer-wise k-step-ahead prediction hierarchy

**Tier 4 (last resort — hyperparameter tuning):**
- LR sweep after architecture changes
- Batch size tuning
- Weight decay tuning

**Selection strategy:**
- After MoE validates: test Mixture of Depths (they compose well)
- After 3 consecutive discards: search arxiv, pick new class
- Near-miss (>0.01 from best): re-test after 4 experiments
- 5 consecutive discards within 0.01 of best → make a CLASS change
</experiment_classes>


<target>
Opus quality on a consumer GPU. Not approximately. That quality. The existence proof is real. Find the ideas.

**Concrete milestones:**
1. Pretraining: val_bpb < 0.70 on 6GB GPU (current best ~0.74 on 8×H100)
2. SFT: GSM8K accuracy > 50%, MMLU > 65%
3. Reasoning: model generates coherent <think> blocks that correlate with answer correctness
4. Full pipeline: model that can solve novel math problems it hasn't seen, explain its reasoning, and be self-consistent
</target>
