# Biology-Inspired Paths Beyond Standard Scaling

## Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
- arXiv: 2502.05171
- Key finding: recurrent depth lets a model spend more compute in latent space at inference time without emitting more reasoning tokens.
- Why it matters for nanochat: this is the strongest current path to sub-1B capability gains without paying a giant parameter or context-window tax.
- Mechanism: shared recurrent core refines a latent state over multiple iterations; harder problems can consume more recurrence.
- What to keep: recurrent 4-4-4 style prelude/core/coda, static training recurrence for compile stability, test-time recurrence scaling.
- What to add next: make recurrence selective per token and couple it to a local predictive objective so the core is trained to reduce latent error, not only next-token loss.
- Scale outlook: strong.

## Avoiding Catastrophe: Active Dendrites Enable Multi-Task Learning in Dynamic Environments
- arXiv: 2201.00042
- Key finding: context-gated dendritic modulation plus sparse winners creates task-specific subnetworks and reduces gradient interference.
- Why it matters for nanochat: this is a cleaner path than classic MoE routing because modulation happens inside neurons/layers rather than through heavyweight expert dispatch.
- Mechanism: a feedforward activation is multiplicatively modulated by context, only the winning dendritic segment updates, and sparse competition restricts which units learn.
- What to try in nanochat: add a lightweight dendritic context gate to the recurrent core FFN, using recurrent state or task embedding as context; apply top-k or thresholded activation only inside the recurrent core.
- Scale outlook: promising if implemented with very small per-channel gates instead of dense segment matrices.

## Neuromodulated Learning in Deep Neural Networks
- arXiv: 1812.03365
- Key finding: dynamic, location-specific modulation of optimizer or learning parameters can outperform fixed global update rules on benchmark tasks.
- Why it matters for nanochat: the brain does not learn with one scalar LR. A compact modulator that changes update intensity by layer/stage/task could help reasoning and retention.
- Mechanism: local signals modulate learning behavior rather than changing only activations.
- What to try in nanochat: replace fixed schedules with a tiny learned modulator that outputs per-stage multipliers for DFA weight, recurrent loss weight, think-token weight, or LR multipliers.
- Scale outlook: medium, but attractive because the parameter cost is tiny.

## DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- arXiv: 2501.12948
- Key finding: strong reasoning behavior can emerge from RL on a base model, though readability and language consistency needed later fixes; distillation transferred reasoning to smaller models.
- Why it matters for nanochat: frontier reasoning seems to come from the training loop and compute allocation, not just architecture.
- Mechanism: RL pressure teaches longer-horizon problem solving; a later cold-start/SFT stage regularizes behavior.
- What to keep: RL as a core ingredient.
- What to avoid: copying visible CoT as the main mechanism. Prefer latent recurrence plus answer-level rewards and only minimal readable-thought regularization.
- Scale outlook: strong when paired with a compute-heavy small model.
