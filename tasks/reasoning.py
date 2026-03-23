"""
Claude Opus 4.6 synthetic reasoning dataset with thinking blocks.
Dataset: Roman1111111/claude-opus-4.6-10000x

The dataset has messages with a separate 'reasoning' field on assistant turns.
We reconstruct assistant content as <think>...</think>\n<answer>...</answer>
so render_conversation_with_think() can assign mask=2 to thinking tokens.

Note: loaded as raw JSON (not via load_dataset metadata) because the dataset's
      metadata uses a 'Json' feature type not supported by older datasets versions.
"""
from datasets import load_dataset
from tasks.common import Task


class OpusReasoning(Task):
    """
    Claude Opus 4.6 reasoning dataset (~9.6K examples).
    90% train / 10% val split applied deterministically after shuffling.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        ds = load_dataset(
            "json",
            data_files="hf://datasets/Roman1111111/claude-opus-4.6-10000x/opus46_final.jsonl",
            split="train",
        )
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
        raw_messages = row.get("messages", [])
        messages = []
        for msg in raw_messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning") or ""
            if role not in ("user", "assistant"):
                continue  # skip system messages
            if role == "assistant" and reasoning:
                # Wrap reasoning in <think> tags so render_conversation_with_think()
                # can identify thinking tokens (mask=2) for phased loss weighting.
                content = f"<think>{reasoning}</think>\n{content}"
            messages.append({"role": role, "content": content})
        if len(messages) < 2:
            return {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        return {"messages": messages}
