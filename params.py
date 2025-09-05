
import torch
from PromptAD import *

model, _, _ = CLIPAD.create_model_and_transforms(model_name="ViT-B-16-plus-240", pretrained="laion400m_e32", precision = "fp16")
PromptLearner = PromptLearner(4, 1, 1, 4, "bagel", model, "fp16")
Missing_PromptLearner = Missing_PromptLearner(36, 6)

params_0 = sum(p.numel() for p in model.visual.parameters())
params_1 = sum(p.numel() for p in model.text.parameters())
params_2 = sum(p.numel() for p in PromptLearner.parameters())
params_3 = sum(p.numel() for p in Missing_PromptLearner.parameters())

print(f"Params of Vision Transformer: {params_0}")
print(f"Params of Text Transformer: {params_1}")
print(f"Params of Antithetical Text Prompt: {params_2}")
print(f"Params of Missing-aware Cross-modal Prompt: {params_3}")