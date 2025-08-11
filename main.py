#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, Any, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Import UI class and configuration
from ui import DebateUI
from config import PRICES, VALIDATION_LIMITS, COST_ESTIMATION

# Constants are now imported from config.py

# ---------- Cost Tracking Class ----------
class CostTracker:
    """Tracks and manages costs for AI model usage."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.cost_history = []  # For future expansion
    
    def add_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Add cost for model usage and return current cost."""
        cost = estimate_cost(model, tokens_in, tokens_out)
        self.total_cost += cost
        self.cost_history.append({
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost
        })
        return cost
    
    def get_total_cost(self) -> float:
        """Get current total cost."""
        return self.total_cost
    
    def reset_cost(self):
        """Reset total cost to zero."""
        self.total_cost = 0.0
        self.cost_history.clear()
    
    def format_cost_message(self) -> str:
        """Format cost message."""
        return f"Total cost: ${self.total_cost:.6f}"


# Create global cost tracker instance
cost_tracker = CostTracker()

# ---------- Enhanced Logging ----------
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("orchestrator")


class LogManager:
    """Centralized logging management with performance tracking."""
    
    def __init__(self, name: str = "debate_system"):
        self.logger = logging.getLogger(name)
        self.operation_stack = []
        self.performance_metrics = {}
    
    def start_operation(self, operation: str, **context):
        """Start logging an operation."""
        operation_id = f"{operation}_{len(self.operation_stack)}"
        self.operation_stack.append({
            "id": operation_id,
            "operation": operation,
            "context": context,
            "start_time": time.time()
        })
        
        self.logger.info(f"START: {operation}", extra={
            "operation_id": operation_id,
            "operation": operation,
            "context": context,
            "action": "start"
        })
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, result: Any = None):
        """End logging an operation."""
        if not self.operation_stack:
            return
        
        op_data = self.operation_stack.pop()
        execution_time = time.time() - op_data["start_time"]
        
        # Store performance metrics
        if op_data["operation"] not in self.performance_metrics:
            self.performance_metrics[op_data["operation"]] = []
        self.performance_metrics[op_data["operation"]].append(execution_time)
        
        if success:
            self.logger.info(f"SUCCESS: {op_data['operation']} in {execution_time:.3f}s", extra={
                "operation_id": operation_id,
                "operation": op_data["operation"],
                "context": op_data["context"],
                "execution_time": execution_time,
                "result": str(result) if result else None,
                "action": "complete"
            })
        else:
            self.logger.error(f"FAILED: {op_data['operation']} after {execution_time:.3f}s", extra={
                "operation_id": operation_id,
                "operation": op_data["operation"],
                "context": op_data["context"],
                "execution_time": execution_time,
                "action": "fail"
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations."""
        summary = {}
        for operation, times in self.performance_metrics.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        return summary


# Create global log manager instance
log_manager = LogManager()


def log_operation(operation_name: str = None, log_args: bool = True, log_result: bool = True):
    """Decorator that logs function entry, exit, and performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            # Start operation logging
            operation_id = log_manager.start_operation(op_name, 
                args=str(args) if log_args else "***",
                kwargs=str(kwargs) if log_args else "***"
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                log_manager.end_operation(operation_id, success=True, 
                    result=str(result) if log_result else "***")
                
                return result
                
            except Exception as e:
                # Log error
                log_manager.end_operation(operation_id, success=False)
                raise
        
        return wrapper
    return decorator


# ---------- Error Handling Decorator ----------
def handle_errors(func):
    """Simple decorator that handles errors and maintains return format."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = func.__name__
            logger.error(f"Error in {context}: {e}")
            
            # Return error in appropriate format based on function
            if func.__name__ == "enhanced_debate_function":
                return f"Error in {context}: {str(e)}", "{}", format_cost_message()
            elif func.__name__ == "estimate_cost_for_question":
                return f"Error in {context}: {str(e)}"
            else:
                return f"Error in {context}: {str(e)}"
    return wrapper


# ---------- Helpers ----------
def get_openai_client() -> OpenAI:
    return OpenAI()


def extract_openai_text(resp: Any) -> str:
    try:
        return resp.output_text.strip()
    except (AttributeError, TypeError, ValueError):
        return ""


# ---------- Validation Functions ----------
def validate_inputs(question: str, max_tokens: int, rounds: int, models: list) -> Tuple[bool, str, Dict[str, Any]]:
    """Universal validation that returns validation result, error message, and validated data."""
    errors = []
    
    # Question validation
    if not question or not question.strip():
        errors.append("Please enter a question.")
    elif len(question.strip()) > VALIDATION_LIMITS["max_question_length"]:
        errors.append(f"Question is too long. Maximum {VALIDATION_LIMITS['max_question_length']} characters allowed.")
    
    # Models validation
    if not models or len(models) < VALIDATION_LIMITS["min_models"]:
        errors.append(f"Please select at least {VALIDATION_LIMITS['min_models']} models (Debater1, Debater2, Judge).")
    
    # Parameters validation
    if max_tokens < VALIDATION_LIMITS["min_tokens"] or max_tokens > VALIDATION_LIMITS["max_tokens"]:
        errors.append(f"Max tokens must be between {VALIDATION_LIMITS['min_tokens']} and {VALIDATION_LIMITS['max_tokens']}.")
    
    if rounds < VALIDATION_LIMITS["min_rounds"] or rounds > VALIDATION_LIMITS["max_rounds"]:
        errors.append(f"Debate rounds must be between {VALIDATION_LIMITS['min_rounds']} and {VALIDATION_LIMITS['max_rounds']}.")
    
    # Return validation result
    is_valid = len(errors) == 0
    error_msg = "; ".join(errors) if errors else ""
    validated_data = {
        "question": question.strip() if question else "",
        "max_tokens": max_tokens,
        "rounds": rounds,
        "models": models
    }
    
    return is_valid, error_msg, validated_data


def format_cost_message() -> str:
    """Centralized cost message formatting."""
    return cost_tracker.format_cost_message()


# ---------- Cost tracking ----------
def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD for given model and token usage."""
    if model not in PRICES:
        return 0.0
    p = PRICES[model]
    return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]


def log_and_add_cost(model: str, tokens_in: int, tokens_out: int):
    """Log cost and update total."""
    cost = cost_tracker.add_cost(model, tokens_in, tokens_out)
    logger.info(f"Model {model} used {tokens_in} in / {tokens_out} out tokens → ${cost:.6f} (total ${cost_tracker.get_total_cost():.6f})")


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
@handle_errors
@log_operation("debate orchestration", log_args=True, log_result=False)
def run_orchestration_with_settings(question: str, max_tokens: int, rounds: int, models: list) -> Tuple[
    str, Dict[str, Any]]:
    """Main orchestration coordinator."""
    oai = get_openai_client()
    
    # Break down into logical blocks:
    debate_rounds = conduct_debate_rounds(question, max_tokens, rounds, models, oai)
    final_answer = get_final_judgment(question, debate_rounds, models[2], max_tokens, oai)
    summary = format_debate_summary(debate_rounds, final_answer, models[2], rounds, models, max_tokens)
    
    return final_answer, summary


@log_operation("debate rounds", log_args=False, log_result=False)
def conduct_debate_rounds(question: str, max_tokens: int, rounds: int, models: list, client: OpenAI) -> list:
    """Conduct debate rounds and return structured data."""
    model_answers = []
    current_question = question
    
    for round_num in range(rounds):
        # Get answers from both debaters
        answer1, _, _ = call_openai_model(
            models[0], current_question, client, max_tokens
        )
        answer2, _, _ = call_openai_model(
            models[1], current_question, client, max_tokens
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
        
        # If not the last round, update question for next iteration
        if round_num < rounds - 1:
            current_question = f"Based on previous answers, continue the debate: {question}"
    
    return model_answers


@log_operation("final judgment", log_args=False, log_result=False)
def get_final_judgment(question: str, debate_rounds: list, judge_model: str, max_tokens: int, client: OpenAI) -> str:
    """Get final judgment from judge model."""
    # Collect all answers for judgment
    answers = []
    for round_data in debate_rounds:
        answers.extend([round_data["debater1"]["answer"], round_data["debater2"]["answer"]])
    
    # Create judge prompt
    judge_prompt = f"""
Compare the {len(answers)} answers below and produce one unified, most accurate answer.

Question: {question}

Answers:
{chr(10).join([f"Answer {i + 1} ({judge_model}): {ans}" for i, ans in enumerate(answers)])}

Rules: Output one answer that is the most correct.
"""
    
    final_answer, _, _ = call_openai_model(
        judge_model, judge_prompt, client, max_tokens
    )
    
    return final_answer


def format_debate_summary(debate_rounds: list, final_answer: str, judge_model: str, rounds: int, models: list, max_tokens: int) -> Dict[str, Any]:
    """Format debate results for display."""
    logger.info("Formatting debate summary")
    
    # Create summary structure
    debate_summary = []
    for round_data in debate_rounds:
        debate_summary.append(f"Round {round_data['round']}:")
        debate_summary.append(f"  {round_data['debater1']['model']}: {round_data['debater1']['answer']}")
        debate_summary.append(f"  {round_data['debater2']['model']}: {round_data['debater2']['answer']}")
        debate_summary.append("")
    
    debate_summary.append(f"Final Judgment ({judge_model}): {final_answer}")
    
    return {
        "summary": "\n".join(debate_summary),
        "rounds": rounds,
        "models": models,
        "max_tokens": max_tokens
    }


# ---------- Enhanced Debate Function ----------
@handle_errors
@log_operation("enhanced debate", log_args=True, log_result=False)
def enhanced_debate_function(question: str, max_tokens: int, rounds: int, cost_lim: float, models: list):
    """Main debate function coordinator."""
    # Validate inputs
    validation_result = validate_debate_inputs(question, max_tokens, rounds, models, cost_lim)
    if not validation_result["is_valid"]:
        return validation_result["error_message"], "{}", format_cost_message()
    
    # Run debate orchestration
    final_answer, meta = run_debate_orchestration(question, max_tokens, rounds, models)
    
    # Format and return results
    return format_debate_results(final_answer, meta)


def validate_debate_inputs(question: str, max_tokens: int, rounds: int, models: list, cost_lim: float) -> Dict[str, Any]:
    """Validate all debate inputs and return validation result."""
    # Universal validation
    is_valid, error_msg, validated_data = validate_inputs(question, max_tokens, rounds, models)
    if not is_valid:
        return {
            "is_valid": False,
            "error_message": f"Error: {error_msg}"
        }
    
    # Cost limit validation
    if cost_lim <= 0:
        return {
            "is_valid": False,
            "error_message": "Error: Cost limit must be greater than 0."
        }
    
    # Environment validation
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "is_valid": False,
            "error_message": "Error: OPENAI_API_KEY environment variable not set"
        }
    
    # Cost limit check
    if cost_tracker.get_total_cost() > cost_lim:
        return {
            "is_valid": False,
            "error_message": f"Cost limit exceeded (${cost_tracker.get_total_cost():.4f} > ${cost_lim:.2f})"
        }
    
    return {
        "is_valid": True,
        "error_message": "",
        "validated_data": validated_data
    }


def run_debate_orchestration(question: str, max_tokens: int, rounds: int, models: list) -> Tuple[str, Dict[str, Any]]:
    """Run the debate orchestration process."""
    logger.info("Starting debate orchestration...")
    
    final_answer, meta = run_orchestration_with_settings(
        question, max_tokens, rounds, models
    )
    
    logger.info("Debate orchestration completed")
    return final_answer, meta


def format_debate_results(final_answer: str, meta: Dict[str, Any]) -> Tuple[str, str, str]:
    """Format debate results for return."""
    logger.info("Formatting debate results...")
    
    # Format meta data for display as text
    try:
        meta_formatted = meta.get("summary", str(meta))
        logger.info("Meta data formatted successfully")
    except Exception as format_error:
        logger.error(f"Meta data formatting error: {format_error}")
        meta_formatted = str(meta)
    
    logger.info("Returning formatted results...")
    return final_answer, meta_formatted, format_cost_message()


# ---------- Cost Estimation Function ----------
@handle_errors
@log_operation("cost estimation", log_args=True, log_result=True)
def estimate_cost_for_question(question: str, max_tokens: int, rounds: int, models: list):
    """Estimate cost for a given question and settings."""
    # Universal validation
    is_valid, error_msg, validated_data = validate_inputs(question, max_tokens, rounds, models)
    if not is_valid:
        return f"Error: {error_msg}"

    # Simple estimation based on question length and settings
    estimated_tokens = len(question.split()) * COST_ESTIMATION["words_per_token"] * rounds * len(models)  # Rough estimate
    estimated_cost = (estimated_tokens / COST_ESTIMATION["tokens_per_million"]) * COST_ESTIMATION["average_cost_per_token"]

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


# ---------- Performance Monitoring Function ----------
def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary for all logged operations."""
    return log_manager.get_performance_summary()


# ---------- Create Enhanced Gradio Interface ----------
def create_interface():
    """Create and configure the enhanced Gradio interface using DebateUI class."""
    
    # Create UI instance
    debate_ui = DebateUI()
    
    # Create interface with event handlers
    demo = debate_ui.create_interface(
        debate_function=enhanced_debate_function,
        cost_estimation_function=estimate_cost_for_question
    )
    
    return demo


# ---------- Main ----------
if __name__ == "__main__":
    # Check environment variables first
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set the OPENAI_API_KEY environment variable before running the app.")
        exit(1)

    # Import configuration
    from config import UI_CONFIG
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name=UI_CONFIG["server_name"],
        server_port=UI_CONFIG["server_port"],
        share=UI_CONFIG["share"],
        show_error=UI_CONFIG["show_error"]
    )
