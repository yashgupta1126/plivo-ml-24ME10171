# Training Run Log

## Run 0 (Baseline)
* **Hypothesis:** Establish baseline performance of the provided mediocre starter code.
* **Changes:** None (Default Adam optimizer, constant LR, ByteTokenizer).
* **Dev BPB:** 2.3718
* **Conclusion:** The baseline is slow and inefficient. Need to implement AdamW and a learning rate scheduler to optimize the limited steps.

## Run 1 (Optimizer Upgrade)
* **Hypothesis:** The constant LR and basic Adam optimizer are inefficient. Adding AdamW, Cosine Annealing with warmup, and gradient clipping will allow better convergence.
* **Changes:** Replaced Adam with AdamW. Added cosine scheduler (100 step warmup). Added `clip_grad_norm_`.
* **Dev BPB:** 2.2516
* **Conclusion:** Massive improvement. However, the raw byte tokenizer is still wasting context length on Hindi text. 

## Run 2 (BPE Tokenizer)
* **Hypothesis:** The raw byte tokenizer uses up to 3 tokens per Devanagari character. Training a custom BPE tokenizer (vocab=512) will compress sequence length and improve efficiency.
* **Changes:** Replaced byte tokenizer with custom BPE in `tokenizer.py`. Vocab size increased to 512. 
* **Dev BPB:** 2.1991
* **Conclusion:** Tokenizer compression successfully lowered BPB.

## Run 3 (Ambitious Architecture Push)
* **Hypothesis:** We have 600k unused parameters. Tying weights and doubling the context window (block_size=256) while pushing the model to 1.92M parameters will yield better context understanding.
* **Changes:** `tie_weights=True`, `n_head=6`, `n_embd=192`, `block_size=256`. 
* **Dev BPB:** 2.2781 (Worse)
* **Conclusion:** The massive model failed to converge. 2,000 steps is too short of a window for 1.92M parameters to settle, resulting in underfitting.

## Run 4 (Aggressive LR on Ambitious Arch)
* **Hypothesis:** A higher learning rate and shorter warmup will force the 1.92M model to learn faster within the 2,000 step limit.
* **Changes:** Increased peak LR to 3e-3, reduced warmup to 50 steps.
* **Dev BPB:** 2.3029 (Worse)
* **Conclusion:** The aggressive LR caused instability.

## Run 5 (The Final Revert & Winning Submission)
* **Hypothesis:** The 1.42M parameter model (Run 2) is the absolute sweet spot for the 2,000-step time limit constraint.
* **Changes:** Reverted entirely back to Run 2's architecture (`tie_weights=False`, 1.42M params) and optimizer settings.
* **Dev BPB:** 2.1991
* **Conclusion:** Re-verified optimal score. Submitted this checkpoint. It proves a smaller, highly-regularized model with compressed vocabulary outperforms an under-trained larger model.