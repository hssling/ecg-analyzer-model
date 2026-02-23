import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import json

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_ID = "hssling/cardioai-adapter"

print("Starting App Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

if ADAPTER_ID:
    print(f"Loading custom fine-tuned LoRA weights: {ADAPTER_ID}")
    try:
        model.load_adapter(ADAPTER_ID)
    except Exception as e:
        print(f"Failed to load adapter. Using base model. Error: {e}")

def diagnose_ecg(image: Image.Image = None, temp: float = 0.4, max_tokens: int = 2000):
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
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=int(max_tokens), temperature=float(temp), top_p=0.9, do_sample=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
        gr.Slider(minimum=256, maximum=4096, value=2000, step=256, label="Max Tokens")
    ],
    outputs=gr.Markdown(label="Clinical Report Output"),
    title="CardioAI Inference API",
    description="Fine-tuned Medical LLM for Electrocardiogram (ECG) Tracings."
)


