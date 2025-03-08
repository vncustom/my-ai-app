import os
import gradio as gr
import requests


API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")
def query(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_response(prompt):
    try:
        output = query({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "return_full_text": False
            }
        })
        return output[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

app = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=5, placeholder="Nhập prompt..."),
    outputs=gr.Textbox(lines=10),
    title="AI Assistant với Qwen2.5-7B",
    examples=["Các kỹ thuật prompt engineering", "Giải thích về mô hình AI"]
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)
