"""
Simple AI Chatbot using Gradio and Phi-2
Trace-free implementation: No complex dependencies for memory
"""
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

# Model
MODEL_ID = "microsoft/phi-2"

# Local Memory Implementation (Replaces LangChain for stability)
class SimpleMemory:
    def __init__(self, k=10):
        self.k = k
        self.messages = []

    def save_context(self, inputs, outputs):
        self.messages.append({"role": "user", "content": inputs["input"]})
        self.messages.append({"role": "assistant", "content": outputs["output"]})
        self._trim()

    def load_memory_variables(self, inputs):
        return {"history": self.messages}

    def clear(self):
        self.messages = []

    def _trim(self):
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2):]

memory = SimpleMemory(k=10)

print("=" * 50)
print("Loading Microsoft Phi-2 model...")
print("=" * 50)

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device.upper()}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

if device == "cpu":
    model = model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")
print("=" * 50)


def build_prompt(message: str, system_prompt: str) -> str:
    """Build prompt with conversation history."""
    prompt = f"Instruct: {system_prompt}\n"
    
    # Get history from simple memory
    history = memory.load_memory_variables({})
    messages = history.get("history", [])
    
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    
    prompt += f"User: {message}\nAssistant:"
    return prompt


def generate_response(message: str, system_prompt: str, max_tokens: int, temperature: float):
    """Generate streaming response."""
    prompt = build_prompt(message, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": int(max_tokens),
        "temperature": float(temperature),
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    response = ""
    for token in streamer:
        response += token
        # Stop at end markers
        if any(m in response for m in ["User:", "Instruct:", "\n\n\n"]):
            break
        yield response
    
    thread.join()
    
    # Clean response
    response = response.split("User:")[0].split("Instruct:")[0].strip()
    
    # Save to memory
    memory.save_context({"input": message}, {"output": response})
    
    yield response


# Gradio UI
with gr.Blocks(title="Simple AI Chatbot") as demo:
    
    gr.Markdown("# 🤖 Simple AI Chatbot (Phi-2)")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500)
            msg = gr.Textbox(label="Message", placeholder="Type here...", lines=2)
            
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            system_prompt = gr.Textbox(
                label="System Prompt",
                value="You are a helpful AI assistant. Give clear and concise answers.",
                lines=3
            )
            max_tokens = gr.Slider(64, 512, value=300, step=16, label="Max Tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            
            gr.Markdown("### Memory")
            memory_status = gr.Textbox(label="Status", value="Ready", interactive=False)
    
    def user_message(message, history, status):
        if not message.strip():
            return "", history, ""
        return "", history + [{"role": "user", "content": message}], ""
    
    def bot_response(history, sys_prompt, max_tok, temp):
        if not history:
            yield history, "No messages"
            return
        
        user_msg = history[-1]["content"]
        history = history + [{"role": "assistant", "content": ""}]
        
        for partial in generate_response(user_msg, sys_prompt, max_tok, temp):
            history[-1]["content"] = partial
            msg_count = len(memory.load_memory_variables({}).get("history", []))
            yield history, f"Memory: {msg_count} messages"
    
    def clear_chat():
        memory.clear()
        return [], "Cleared!"
    
    send_btn.click(
        user_message, [msg, chatbot, memory_status], [msg, chatbot, memory_status]
    ).then(
        bot_response, [chatbot, system_prompt, max_tokens, temperature], [chatbot, memory_status]
    )
    
    msg.submit(
        user_message, [msg, chatbot, memory_status], [msg, chatbot, memory_status]
    ).then(
        bot_response, [chatbot, system_prompt, max_tokens, temperature], [chatbot, memory_status]
    )
    
    clear_btn.click(clear_chat, None, [chatbot, memory_status])

if __name__ == "__main__":

    demo.queue().launch()
