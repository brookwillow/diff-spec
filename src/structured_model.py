from __future__ import annotations

from typing import Any


def build_structured_model(base_model_name: str, label_sizes: dict[str, int]):
    import torch
    from torch import nn
    from transformers import AutoModel, PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class StructuredDrafterModel(PreTrainedModel):
        def __init__(self):
            encoder = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
            super().__init__(encoder.config)
            self.encoder = encoder
            hidden_size = getattr(encoder.config, "hidden_size")
            self.kind_head = nn.Linear(hidden_size, label_sizes["kind"])
            self.tool_head = nn.Linear(hidden_size, label_sizes["tool"])
            self.slot_heads = nn.ModuleDict(
                {name: nn.Linear(hidden_size, size) for name, size in label_sizes.items() if name not in {"kind", "tool"}}
            )
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            kind_labels=None,
            tool_labels=None,
            **slot_labels: Any,
        ):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

            logits = {
                "kind": self.kind_head(pooled),
                "tool": self.tool_head(pooled),
            }
            logits.update({name: head(pooled) for name, head in self.slot_heads.items()})

            loss = None
            losses = []
            if kind_labels is not None:
                losses.append(self.loss_fn(logits["kind"], kind_labels))
            if tool_labels is not None:
                losses.append(self.loss_fn(logits["tool"], tool_labels))
            for name in self.slot_heads:
                label = slot_labels.get(f"{name}_labels")
                if label is not None:
                    losses.append(self.loss_fn(logits[name], label))
            if losses:
                loss = torch.stack(losses).mean()
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return StructuredDrafterModel()
