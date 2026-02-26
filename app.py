import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import json
import os

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_ADAPTER_ID = "hssling/cardioai-adapter"
CONFIG_PATH = "model_config.json"

def load_runtime_config():
    config = {
        "base_model": os.environ.get("BASE_MODEL_ID", DEFAULT_MODEL_ID),
        "adapter_repo": os.environ.get("ADAPTER_REPO_ID", DEFAULT_ADAPTER_ID),
        "adapter_revision": os.environ.get("ADAPTER_REVISION", "main")
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                disk_cfg = json.load(f)
            config["base_model"] = disk_cfg.get("base_model", config["base_model"])
            config["adapter_repo"] = disk_cfg.get("adapter_repo", config["adapter_repo"])
            config["adapter_revision"] = disk_cfg.get("adapter_revision", config["adapter_revision"])
        except Exception as e:
            print(f"Failed to read {CONFIG_PATH}; falling back to defaults. Error: {e}")
    return config

cfg = load_runtime_config()
MODEL_ID = cfg["base_model"]
ADAPTER_ID = cfg["adapter_repo"]
ADAPTER_REV = cfg["adapter_revision"]

print("Starting App Engine...")
os.makedirs("/tmp/offload", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

model_kwargs = {
    "pretrained_model_name_or_path": MODEL_ID,
    "device_map": "auto",
    "low_cpu_mem_usage": True,
    "offload_folder": "/tmp/offload"
}

if device == "cuda":
    model_kwargs["torch_dtype"] = torch.float16
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
else:
    # CPU space: keep dtype low to reduce memory footprint.
    model_kwargs["torch_dtype"] = torch.float16

model = Qwen2VLForConditionalGeneration.from_pretrained(**model_kwargs)

if ADAPTER_ID:
    print(f"Loading custom fine-tuned LoRA weights: {ADAPTER_ID}@{ADAPTER_REV}")
    try:
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_ID,
            revision=ADAPTER_REV,
            is_trainable=False
        )
        print("Adapter load successful.")
    except Exception as e:
        print(f"Failed to load adapter; serving base model instead. Error: {e}")

def diagnose_ecg(image: Image.Image = None, temp: float = 0.4, max_tokens: int = 768):
    try:
        if image is None:
            return json.dumps({"error": "No image provided."})

        system_prompt = "You are CardioAI, a highly advanced expert Cardiologist. Analyze the provided Electrocardiogram (ECG/EKG)."
        user_prompt = "Analyze this 12-lead Electrocardiogram trace and extract the detailed clinical rhythms and pathological findings in a structured format."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        model_device = model.device if hasattr(model, "device") else torch.device(device)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=int(max_tokens), temperature=float(temp), top_p=0.9, do_sample=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return output_text

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=diagnose_ecg,
    inputs=[
        gr.Image(type="pil", label="ECG Image Scan"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, label="Temperature"),
        gr.Slider(minimum=128, maximum=1536, value=768, step=128, label="Max Tokens")
    ],
    outputs=gr.Markdown(label="Clinical Report Output"),
    title="CardioAI Inference API",
    description="Fine-tuned Medical LLM for Electrocardiogram (ECG) Tracings."
)

if __name__ == "__main__":
    demo.launch()
