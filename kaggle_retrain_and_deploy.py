# %% [markdown]
# CardioAI Kaggle Notebook: Retrain + Deploy to Hugging Face Space
#
# This script is notebook-friendly (run cell by cell in Kaggle).
# Outcome:
# 1) Fine-tune LoRA adapter on ECG image dataset.
# 2) Push adapter to HF model repo.
# 3) Auto-update HF Space config so app serves the new adapter revision.

# %% Install deps (run once in a Kaggle cell)
# !pip -q install -U "transformers>=4.49.0" "datasets>=2.19.0" "accelerate>=0.34.0" "peft>=0.13.0" "huggingface_hub>=0.26.0" "Pillow>=10.0.0"
# !pip -q install -U "bitsandbytes>=0.46.1"
# # After installing/upgrading bitsandbytes on Kaggle, restart session once, then run all cells.

# %%
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# %%
# ----------------------------
# CONFIG (edit these values)
# ----------------------------
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_ID = "IdaFLab/ECG-Plot-Images"  # Suitable ECG plot dataset used in your current pipeline
DATASET_SPLIT = "train[:3000]"          # Raise when stable (e.g. full train)

HF_ADAPTER_REPO = "hssling/cardioai-adapter"
HF_SPACE_REPO = "hssling/cardioai-api"  # Space repo to auto-point to newest adapter revision

OUTPUT_DIR = "/kaggle/working/cardioai_adapter"
SEED = 42

EPOCHS = 2
LR = 2e-4
TRAIN_BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_TOKENS = 768
LOAD_IN_4BIT = True

# %%
# ----------------------------
# Auth from Kaggle Secrets
# ----------------------------
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    HF_TOKEN = _secrets.get_secret("HF_TOKEN")
except Exception as e:
    raise RuntimeError("Missing Kaggle secret HF_TOKEN") from e

os.environ["HF_TOKEN"] = HF_TOKEN
api = HfApi(token=HF_TOKEN)

random.seed(SEED)
torch.manual_seed(SEED)

print("Authenticated to Hugging Face Hub.")

# %%
def has_compatible_bitsandbytes() -> bool:
    try:
        import bitsandbytes as bnb  # type: ignore
        ver = getattr(bnb, "__version__", "0.0.0")
        major, minor, patch = [int(x) for x in ver.split(".")[:3]]
        return (major, minor, patch) >= (0, 46, 1)
    except Exception:
        return False

# %%
# ----------------------------
# Load processor + base model
# ----------------------------
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN, use_fast=False)

use_4bit_now = LOAD_IN_4BIT and torch.cuda.is_available() and has_compatible_bitsandbytes()
if use_4bit_now:
    print("Using 4-bit quantization with bitsandbytes.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
else:
    print("bitsandbytes>=0.46.1 not available (or no CUDA). Falling back to fp16/bf16 load.")
    bnb_config = None

model_kwargs = {
    "pretrained_model_name_or_path": BASE_MODEL_ID,
    "device_map": "auto",
    "token": HF_TOKEN
}
if use_4bit_now:
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["torch_dtype"] = torch.float16
else:
    model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = Qwen2VLForConditionalGeneration.from_pretrained(**model_kwargs)
if use_4bit_now:
    model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# %%
# ----------------------------
# Dataset and formatting
# ----------------------------
dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
dataset = dataset.shuffle(seed=SEED)

label_map = {
    0: "Normal sinus rhythm with no significant ectopy.",
    1: "Supraventricular ectopic activity is present.",
    2: "Ventricular ectopic beats are present.",
    3: "Fusion beat pattern is present."
}

def to_train_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    # Keep mapping stable with your existing dataset schema.
    finding = label_map.get(int(ex.get("type", 0)), "ECG abnormality present; clinical correlation advised.")

    messages = [
        {
            "role": "system",
            "content": "You are CardioAI, an expert cardiology assistant for ECG interpretation."
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this ECG and provide rhythm, key abnormalities, and a short impression."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": finding}]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"image": ex["image"], "text": text}

train_ds = dataset.map(to_train_example, remove_columns=dataset.column_names)

@dataclass
class ECGCollator:
    processor: Any
    max_tokens: int = MAX_TOKENS

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [x["image"].convert("RGB") for x in batch]
        texts = [x["text"] for x in batch]
        model_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens
        )
        labels = model_inputs["input_ids"].clone()
        # Ignore padding in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

collator = ECGCollator(processor=processor)

# %%
# ----------------------------
# Train
# ----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=20,
    save_strategy="epoch",
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator
)

trainer.train()

# %%
# ----------------------------
# Save + Push adapter
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)   # PEFT adapter files
processor.save_pretrained(OUTPUT_DIR)

api.create_repo(HF_ADAPTER_REPO, repo_type="model", exist_ok=True)
commit_info = api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=HF_ADAPTER_REPO,
    repo_type="model",
    commit_message="Kaggle retrain: refresh ECG LoRA adapter"
)

if hasattr(commit_info, "oid"):
    adapter_revision = commit_info.oid
else:
    adapter_revision = "main"

print(f"Adapter pushed: https://huggingface.co/{HF_ADAPTER_REPO}")
print(f"Adapter revision: {adapter_revision}")

# %%
# ----------------------------
# Update Space runtime config
# ----------------------------
space_cfg = {
    "base_model": BASE_MODEL_ID,
    "adapter_repo": HF_ADAPTER_REPO,
    "adapter_revision": adapter_revision
}

api.upload_file(
    path_or_fileobj=json.dumps(space_cfg, indent=2).encode("utf-8"),
    path_in_repo="model_config.json",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    commit_message=f"Point space to adapter revision {adapter_revision}"
)

try:
    api.restart_space(repo_id=HF_SPACE_REPO)
    print("Space restart requested.")
except Exception as e:
    print(f"Space restart API call failed (manual restart may be needed): {e}")

print(f"Space URL: https://huggingface.co/spaces/{HF_SPACE_REPO}")
print("Done. Your app can continue using the same Space endpoint.")
