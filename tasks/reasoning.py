"""
Claude Opus 4.6 synthetic reasoning dataset with thinking blocks.
Dataset: Roman1111111/claude-opus-4.6-10000x

Assistant responses contain <think>...</think> blocks before the final answer.
These are handled by render_conversation_with_think() in the tokenizer,
which assigns mask=2 to thinking tokens for phased loss weighting during SFT.
"""
from datasets import load_dataset
from tasks.common import Task


class OpusReasoning(Task):
    """
    Claude Opus 4.6 reasoning dataset (~10K examples).
    90% train / 10% val split applied deterministically after shuffling.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        ds = load_dataset("Roman1111111/claude-opus-4.6-10000x", split="train")
        ds = ds.shuffle(seed=42)
        n = len(ds)
        split_idx = int(0.9 * n)
        if split == "train":
            self.ds = ds.select(range(split_idx))
        else:
            self.ds = ds.select(range(split_idx, n))
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        # Handle common dataset schemas
        if "messages" in row:
            messages = row["messages"]
        elif "conversations" in row:
            messages = [
                {
                    "role": "user" if m.get("from", m.get("role", "")) in ("human", "user") else "assistant",
                    "content": m.get("value", m.get("content", "")),
                }
                for m in row["conversations"]
            ]
        elif "prompt" in row and "response" in row:
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
        else:
            # Unknown schema — return a safe no-op conversation
            return {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        return {"messages": messages}
