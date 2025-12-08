from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
from transformers import pipeline
from gtts import gTTS
import torch
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_device('cpu')

app = FastAPI()

# Load Whisper model on CPU
print("Loading Whisper model on CPU...")
asr_model = whisper.load_model("small", device="cpu")
print("Whisper loaded!")

# Load LLM model on CPU
print("Loading LLM model on CPU (this may take a few minutes)...")
llm = pipeline("text-generation", 
               model="meta-llama/Llama-3.2-1B-Instruct",
               device="cpu",
               torch_dtype=torch.float32)
print("LLM loaded!")

conversation_history = []

# ============================================================
# WEEK 6 ADDITIONS: TOOL FUNCTIONS
# ============================================================

def search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search or return a dummy passage for the given query.
    In a real system, this might query the arXiv API and extract a summary.
    """
    # Placeholder implementation with realistic-looking responses
    arxiv_results = {
        "quantum": "Found paper: 'Quantum Entanglement in Multi-Party Systems' by Smith et al. (2024). Abstract: This paper explores the dynamics of quantum entanglement across multiple particles, demonstrating novel approaches to maintaining coherence in noisy environments.",
        "deep learning": "Found paper: 'Advances in Deep Neural Networks' by Johnson et al. (2024). Abstract: We present new techniques for training deep neural networks with improved generalization and reduced computational requirements.",
        "machine learning": "Found paper: 'Machine Learning for Scientific Discovery' by Lee et al. (2024). Abstract: This work demonstrates how ML can accelerate scientific research across various domains.",
        "neural network": "Found paper: 'Efficient Neural Network Architectures' by Chen et al. (2024). Abstract: We introduce optimized architectures that achieve state-of-the-art performance with fewer parameters.",
    }
    
    # Simple keyword matching for demo purposes
    query_lower = query.lower()
    for keyword, result in arxiv_results.items():
        if keyword in query_lower:
            return result
    
    # Default response if no keyword match
    return f"[arXiv search results for '{query}': Found several papers related to {query}. The most recent work discusses novel approaches and methodologies in this field.]"


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    Uses sympy for safe evaluation.
    """
    try:
        from sympy import sympify
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {e}"


# ============================================================
# WEEK 6: FUNCTION ROUTING LOGIC
# ============================================================

def route_llm_output(llm_output: str) -> str:
    """
    Route LLM response to the correct tool if it's a function call, else return the text.
    Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
    """
    # Clean up the output - sometimes LLMs add extra text
    llm_output = llm_output.strip()
    
    # NEW: Take only the first line if multiple lines exist
    first_line = llm_output.split('\n')[0].strip()
    
    # Try to find JSON in the output
    try:
        # Look for JSON object in the text
        start_idx = first_line.find('{')  # Changed from llm_output to first_line
        end_idx = first_line.rfind('}')   # Changed from llm_output to first_line
        
        if start_idx != -1 and end_idx != -1:
            json_str = first_line[start_idx:end_idx+1]  # Changed
            output = json.loads(json_str)
            func_name = output.get("function")
            args = output.get("arguments", {})
            
            # Route to appropriate function
            if func_name == "search_arxiv":
                query = args.get("query", "")
                result = search_arxiv(query)
                print(f"[FUNCTION CALL] search_arxiv('{query}') -> {result[:100]}...")
                return result
            elif func_name == "calculate":
                expr = args.get("expression", "")
                result = calculate(expr)
                print(f"[FUNCTION CALL] calculate('{expr}') -> {result}")
                return result
            else:
                print(f"[WARNING] Unknown function: {func_name}")
                return f"I tried to call a function '{func_name}' but it's not available."
        else:
            # No JSON found, return as regular text
            return llm_output
            
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        # Not a JSON function call; return the text directly
        print(f"[INFO] Not a function call, returning text response")
        return llm_output


# ============================================================
# MODIFIED WEEK 3 FUNCTIONS WITH FUNCTION CALLING
# ============================================================

def transcribe_audio(audio_bytes):
    """Convert audio bytes to text using Whisper"""
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    
    result = asr_model.transcribe("temp.wav")
    return result["text"]


def generate_response(user_text):
    """
    Generate a response from the LLM, which may be a function call or regular text.
    WEEK 6 MODIFICATION: Added system prompt for function calling.
    """
    conversation_history.append({"role": "user", "text": user_text})
    
    # WEEK 6: System prompt that teaches the model about function calling
    system_prompt = """You are a helpful AI assistant.

You have TWO special functions:
1. calculate - ONLY for math expressions
2. search_arxiv - ONLY for finding academic papers

CRITICAL: Use functions ONLY for math/research. Everything else gets normal text responses.

CORRECT examples:
User: "What is 50 divided by 2?"
Assistant: {"function": "calculate", "arguments": {"expression": "50/2"}}

User: "Search for deep learning papers"
Assistant: {"function": "search_arxiv", "arguments": {"query": "deep learning"}}

User: "Hello!"
Assistant: Hi there! How can I help you today?

User: "How are you?"
Assistant: I'm doing great, thank you! I'm here to help with math calculations and research paper searches.

User: "What's your favorite color?"
Assistant: I don't have personal preferences, but I'd be happy to help you with calculations or finding research papers!

User: "Calculate sqrt(144)"
Assistant: {"function": "calculate", "arguments": {"expression": "sqrt(144)"}}

REMEMBER: Only output JSON for math and research. Everything else is normal conversation.
"""
    
    # Build the prompt with conversation history
    prompt = system_prompt + "\n"
    for turn in conversation_history[-5:]:  # Keep last 5 turns for context
        prompt += f"{turn['role']}: {turn['text']}\n"
    
    prompt += "assistant:"
    
    # Generate response from LLM
    outputs = llm(prompt, max_new_tokens=50, do_sample=True, temperature=0.5)
    generated_text = outputs[0]["generated_text"]
    
    # Extract only the new response (remove the prompt)
    llm_response = generated_text[len(prompt):].strip()
    
    if not llm_response:
        llm_response = "I understand."
    
    print(f"[LLM RAW OUTPUT] {llm_response}")
    
    # WEEK 6: Route through function calling logic
    bot_response = route_llm_output(llm_response)
    
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response


def synthesize_speech(bot_text):
    """Convert text to speech using gTTS"""
    output_path = "response.wav"
    tts = gTTS(text=bot_text, lang='en', slow=False)
    tts.save(output_path)
    return output_path


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Week 6 Voice Agent with Function Calling",
        "features": [
            "Voice input/output",
            "Multi-turn conversation",
            "arXiv paper search",
            "Mathematical calculations"
        ],
        "docs": "/docs"
    }


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    """
    Main chat endpoint: receives audio, transcribes it, generates a response
    (possibly calling tools), and returns audio response.
    """
    audio_bytes = await file.read()
    
    # Step 1: Transcribe audio to text
    user_text = transcribe_audio(audio_bytes)
    print(f"\n{'='*60}")
    print(f"[USER SAID] {user_text}")
    
    # Step 2: Generate response (may involve function calling)
    bot_text = generate_response(user_text)
    print(f"[BOT RESPONSE] {bot_text}")
    
    # Step 3: Convert response to speech
    audio_path = synthesize_speech(bot_text)
    print(f"[AUDIO GENERATED] {audio_path}")
    print(f"{'='*60}\n")
    
    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/text-chat/")
async def text_chat_endpoint(request: dict):
    """
    Text-only endpoint for testing without audio files.
    Useful for debugging and testing function calls.
    """
    user_text = request.get("text", "")
    
    print(f"\n{'='*60}")
    print(f"[USER TEXT] {user_text}")
    
    bot_text = generate_response(user_text)
    print(f"[BOT RESPONSE] {bot_text}")
    print(f"{'='*60}\n")
    
    return {"response": bot_text}


@app.post("/reset/")
async def reset_conversation():
    """Reset the conversation history"""
    global conversation_history
    conversation_history = []
    return {"message": "Conversation history cleared"}


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    import sys
    
    print("\n" + "="*60)
    print("Week 6 Voice Agent with Function Calling")
    print("="*60)
    print("\nStarting server...")
    print("API will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("\nAvailable endpoints:")
    print("  POST /chat/          - Voice input/output")
    print("  POST /text-chat/     - Text input/output (for testing)")
    print("  POST /reset/         - Clear conversation history")
    print("="*60 + "\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!\n")
        sys.exit(0)
