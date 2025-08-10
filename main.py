#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Dict, Any, Tuple
import gradio as gr
from dotenv import load_dotenv

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ---------- Config ----------
DEBATER1_MODEL = "gpt-5-mini"       # First debater (OpenAI, smarter)
DEBATER2_MODEL = "gpt-5-nano"       # Second debater (OpenAI, cheaper/faster)
JUDGE_MODEL = "gpt-4o-mini"         # Judge (OpenAI)

# Available models for selection
AVAILABLE_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano", 
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301"
]

# Prices per 1M tokens
PRICES = {
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 0.30, "out": 1.20},
    "gpt-4-turbo": {"in": 0.01, "out": 0.03},
    "gpt-4": {"in": 0.03, "out": 0.06},
    "gpt-3.5-turbo": {"in": 0.0015, "out": 0.002},
    "gpt-3.5-turbo-16k": {"in": 0.003, "out": 0.004},
    "gpt-4-turbo-preview": {"in": 0.01, "out": 0.03},
    "gpt-4-1106-preview": {"in": 0.01, "out": 0.03},
    "gpt-4-0613": {"in": 0.03, "out": 0.06},
    "gpt-4-32k": {"in": 0.06, "out": 0.12},
    "gpt-4-32k-0613": {"in": 0.06, "out": 0.12},
    "gpt-3.5-turbo-0613": {"in": 0.0015, "out": 0.002},
    "gpt-3.5-turbo-0301": {"in": 0.0015, "out": 0.002}
}

total_cost = 0.0  # global accumulator

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("orchestrator")


# ---------- Helpers ----------
def get_openai_client() -> OpenAI:
    return OpenAI()


def extract_openai_text(resp: Any) -> str:
    try:
        return resp.output_text.strip()
    except Exception:
        return ""


# ---------- Validation Functions ----------
def validate_inputs(question: str, max_tokens: int, rounds: int, models: list) -> Tuple[bool, str]:
    """Centralized validation for all inputs."""
    if not question or not question.strip():
        return False, "Error: Please enter a question."
    
    if len(question.strip()) > 1000:
        return False, "Error: Question is too long. Maximum 1000 characters allowed."
    
    if not models or len(models) < 3:
        return False, "Error: Please select at least 3 models (Debater1, Debater2, Judge)."
    
    if max_tokens < 100 or max_tokens > 4000:
        return False, "Error: Max tokens must be between 100 and 4000."
    
    if rounds < 1 or rounds > 3:
        return False, "Error: Debate rounds must be between 1 and 3."
    
    return True, ""


def format_cost_message() -> str:
    """Centralized cost message formatting."""
    return f"Total cost: ${total_cost:.6f}"


# ---------- Cost tracking ----------
def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD for given model and token usage."""
    if model not in PRICES:
        return 0.0
    p = PRICES[model]
    return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]


def log_and_add_cost(model: str, tokens_in: int, tokens_out: int):
    """Log cost and update total."""
    global total_cost
    cost = estimate_cost(model, tokens_in, tokens_out)
    total_cost += cost
    logger.info(f"Model {model} used {tokens_in} in / {tokens_out} out tokens → ${cost:.6f} (total ${total_cost:.6f})")


# ---------- LLM calls ----------
def call_openai_model(model: str, prompt: str, client: OpenAI, max_tokens: int) -> Tuple[str, int, int]:
    """Call OpenAI model and return (text, tokens_in, tokens_out)."""
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=max_tokens,
    )
    txt = extract_openai_text(resp)
    usage = getattr(resp, "usage", None)
    tokens_in = usage.input_tokens if usage else len(prompt.split()) * 1.3
    tokens_out = usage.output_tokens if usage else len(txt.split()) * 1.3
    log_and_add_cost(model, tokens_in, tokens_out)
    return txt, tokens_in, tokens_out


# ---------- Enhanced Orchestration ----------
def run_orchestration_with_settings(question: str, max_tokens: int, rounds: int, models: list) -> Tuple[str, Dict[str, Any]]:
    """Enhanced orchestration with configurable parameters."""
    logger.info(f"Step 1: Initialize client with {rounds} rounds")
    oai = get_openai_client()
    
    answers = []
    model_answers = []  # Store answers with model names
    
    for round_num in range(rounds):
        logger.info(f"Round {round_num + 1}/{rounds}")
        
        # Get answers with fixed temperature settings
        answer1, _, _ = call_openai_model(
            models[0], question, oai, max_tokens
        )
        answer2, _, _ = call_openai_model(
            models[1], question, oai, max_tokens
        )
        
        # Store answers with model names
        model_answers.append({
            "round": round_num + 1,
            "debater1": {
                "model": models[0],
                "answer": answer1
            },
            "debater2": {
                "model": models[1],
                "answer": answer2
            }
        })
        
        answers.extend([answer1, answer2])
        
        # If not the last round, use answers for next iteration
        if round_num < rounds - 1:
            question = f"Based on previous answers, continue the debate: {question}"
    
    # Final judgment
    judge_prompt = f"""
Compare the {len(answers)} answers below and produce one unified, most accurate answer.

Question: {question}

Answers:
{chr(10).join([f"Answer {i+1} ({models[i % 2]}): {ans}" for i, ans in enumerate(answers)])}

Rules: Output one answer that is the most correct.
"""
    
    final_answer, _, _ = call_openai_model(
        models[2], judge_prompt, oai, max_tokens
    )
    
    # Simplify the structure to avoid Gradio compatibility issues
    debate_summary = []
    for round_data in model_answers:
        debate_summary.append(f"Round {round_data['round']}:")
        debate_summary.append(f"  {round_data['debater1']['model']}: {round_data['debater1']['answer'][:200]}...")
        debate_summary.append(f"  {round_data['debater2']['model']}: {round_data['debater2']['answer'][:200]}...")
        debate_summary.append("")
    
    debate_summary.append(f"Final Judgment ({models[2]}): {final_answer[:300]}...")
    
    return final_answer, {
        "summary": "\n".join(debate_summary),
        "rounds": rounds,
        "models": models,
        "max_tokens": max_tokens
    }


# ---------- Enhanced Debate Function ----------
def enhanced_debate_function(question: str, max_tokens: int, rounds: int, cost_lim: float, models: list):
    """Enhanced debate function with all settings."""
    # Centralized validation
    is_valid, error_msg = validate_inputs(question, max_tokens, rounds, models)
    if not is_valid:
        return error_msg, "{}", format_cost_message()
    
    if cost_lim <= 0:
        return "Error: Cost limit must be greater than 0.", "{}", format_cost_message()
    
    try:
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            return "Error: OPENAI_API_KEY environment variable not set", "{}", format_cost_message()
        
        # Check cost limit
        if total_cost > cost_lim:
            return f"Cost limit exceeded (${total_cost:.4f} > ${cost_lim:.2f})", "{}", format_cost_message()
        
        # Run enhanced orchestration
        logger.info("Starting orchestration...")
        final_answer, meta = run_orchestration_with_settings(
            question, max_tokens, rounds, models
        )
        logger.info("Orchestration completed, formatting meta data...")
        
        # Format meta data for display as text
        try:
            meta_formatted = meta.get("summary", str(meta))
            logger.info("Meta data formatted successfully")
        except Exception as format_error:
            logger.error(f"Meta data formatting error: {format_error}")
            meta_formatted = str(meta)
        
        logger.info("Returning results...")
        return final_answer, meta_formatted, format_cost_message()
        
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        return error_msg, "{}", format_cost_message()


# ---------- Cost Estimation Function ----------
def estimate_cost_for_question(question: str, max_tokens: int, rounds: int, models: list):
    """Estimate cost for a given question and settings."""
    # Reuse validation logic
    is_valid, error_msg = validate_inputs(question, max_tokens, rounds, models)
    if not is_valid:
        return error_msg
    
    # Simple estimation based on question length and settings
    estimated_tokens = len(question.split()) * 3 * rounds * len(models)  # Rough estimate
    estimated_cost = (estimated_tokens / 1_000_000) * 0.25  # Average cost per token
    
    return f"💰 Estimated cost: ${estimated_cost:.4f} (based on ~{estimated_tokens} tokens)"


# ---------- Settings Update Function ----------
def update_settings_display(max_tokens, rounds, cost_lim, models):
    """Update the settings display."""
    return {
        "max_tokens": max_tokens,
        "debate_rounds": rounds,
        "cost_limit": cost_lim,
        "selected_models": models
    }


# ---------- Create Enhanced Gradio Interface ----------
def create_interface():
    """Create and configure the enhanced Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .settings-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .char-counter {
        font-size: 12px;
        color: #666;
        text-align: right;
        margin-top: 5px;
    }
    .preset-buttons {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    .preset-btn {
        flex: 1;
    }
    """
    
    with gr.Blocks(css=css, title="AI Debate System", theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1 class='main-header'>🤖 AI Debate System</h1>")
        gr.HTML("<p style='text-align: center; color: #7f8c8d;'>Get AI-powered answers through intelligent debate and judgment</p>")
        
        # === ADVANCED SETTINGS SECTION ===
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h4>Model Parameters:</h4>")
                    
                    # Max tokens setting
                    max_tokens_slider = gr.Slider(
                        minimum=100, maximum=4000, value=1200, step=100,
                        label="Max Output Tokens",
                        info="Maximum tokens for each model response"
                    )
                    
                    # Debate rounds setting
                    debate_rounds = gr.Slider(
                        minimum=1, maximum=3, value=1, step=1,
                        label="Debate Rounds",
                        info="Number of debate rounds (1-3)"
                    )
                
                with gr.Column():
                    gr.HTML("<h4>Model Selection:</h4>")
                    
                    # Model selection
                    model_selector = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value=[DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL],
                        label="Select Models (Debater1, Debater2, Judge)",
                        multiselect=True,
                        info="Choose which models to use"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h4>Cost Control:</h4>")
                    
                    # Cost limit
                    cost_limit = gr.Number(
                        value=0.10, precision=2,
                        label="Cost Limit ($)",
                        info="Stop if total cost exceeds this amount"
                    )
                    
                    # Cost estimation button
                    estimate_btn = gr.Button("💰 Estimate Cost", variant="secondary")
                    cost_estimate = gr.Textbox(
                        label="Estimated Cost",
                        interactive=False
                    )
                
                with gr.Column():
                    gr.HTML("<h4>Quick Presets:</h4>")
            
            with gr.Row():
                fast_btn = gr.Button("🚀 Fast", variant="secondary", size="sm", elem_classes=["preset-btn"])
                quality_btn = gr.Button("⭐ Quality", variant="secondary", size="sm", elem_classes=["preset-btn"])
                balanced_btn = gr.Button("⚖️ Balanced", variant="secondary", size="sm", elem_classes=["preset-btn"])
        
        # === MAIN INTERFACE ===
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Enter your question:",
                    placeholder="Type your question here... (max 1000 characters)",
                    lines=5,
                    max_lines=10,
                    max_length=1000
                )
                
                # Character counter
                char_counter = gr.HTML(
                    value="<div class='char-counter'>Characters: 0/1000</div>",
                    label=""
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Get AI Debate Answer", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear All", variant="secondary", size="lg")
            
            with gr.Column(scale=1):
                
                # Wrap settings display in collapsible accordion
                with gr.Accordion("Active Configuration", open=False):
                    settings_display = gr.JSON(
                        value={}
                    )
        
        # === OUTPUT SECTION ===
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr.HTML("<h4>Final Answer</h4>")
                    copy_btn = gr.Button("📋 Copy", variant="secondary", size="sm")
                
                final_answer_output = gr.Textbox(
                    label="",
                    lines=8,
                    max_lines=15
                )
            
            with gr.Column():
                        meta_output = gr.Textbox(
            label="Debate Details & Individual AI Responses",
            lines=15,
            max_lines=20,
            interactive=False
        )
        
        cost_output = gr.Textbox(
            label="Cost Information"
        )
        
        # === EVENT HANDLERS ===
        # Update character counter with max length
        def update_char_counter(text):
            return f"<div class='char-counter'>Characters: {len(text)}/1000</div>"
        
        question_input.change(fn=update_char_counter, inputs=[question_input], outputs=[char_counter])
        
        # Clear all fields
        def clear_all():
            return "", "", "", "", {}
        
        clear_btn.click(
            fn=clear_all,
            outputs=[question_input, final_answer_output, cost_output, char_counter, settings_display]
        )
        
        # Copy answer to clipboard
        def copy_to_clipboard(text):
            return gr.update(value=text)
        
        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[final_answer_output],
            outputs=[question_input]  # Временно показываем в поле ввода для демонстрации
        )
        
        # Preset functions
        def apply_fast_preset():
            return 800, 1, 0.05, [DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL]
        
        def apply_quality_preset():
            return 2000, 2, 0.20, [DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL]
        
        def apply_balanced_preset():
            return 1200, 1, 0.10, [DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL]
        
        # Apply presets
        fast_btn.click(
            fn=apply_fast_preset,
            outputs=[max_tokens_slider, debate_rounds, cost_limit, model_selector]
        )
        
        quality_btn.click(
            fn=apply_quality_preset,
            outputs=[max_tokens_slider, debate_rounds, cost_limit, model_selector]
        )
        
        balanced_btn.click(
            fn=apply_balanced_preset,
            outputs=[max_tokens_slider, debate_rounds, cost_limit, model_selector]
        )
        
        # Update settings display when parameters change - consolidated
        def update_settings_wrapper(*args):
            return update_settings_display(*args)
        
        for component in [max_tokens_slider, debate_rounds, cost_limit, model_selector]:
            component.change(
                fn=update_settings_wrapper,
                inputs=[max_tokens_slider, debate_rounds, cost_limit, model_selector],
                outputs=settings_display
            )
        
        # Cost estimation button
        estimate_btn.click(
            fn=estimate_cost_for_question,
            inputs=[question_input, max_tokens_slider, debate_rounds, model_selector],
            outputs=[cost_estimate]
        )
        
        # Main submit button with enhanced function
        submit_btn.click(
            fn=enhanced_debate_function,
            inputs=[
                question_input, max_tokens_slider, debate_rounds,
                cost_limit, model_selector
            ],
            outputs=[final_answer_output, meta_output, cost_output]
        )
        
        # Add example questions
        gr.Examples(
            examples=[
                "What are the main differences between Python and JavaScript?",
                "Explain the concept of machine learning in simple terms",
                "What are the pros and cons of remote work?",
                "How does photosynthesis work?"
            ],
            inputs=question_input
        )
        
        # Initialize settings display
        demo.load(fn=update_settings_display, 
                 inputs=[max_tokens_slider, debate_rounds, cost_limit, model_selector], 
                 outputs=settings_display)
    
    return demo


# ---------- Main ----------
if __name__ == "__main__":
    # Check environment variables first
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set the OPENAI_API_KEY environment variable before running the app.")
        exit(1)
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
