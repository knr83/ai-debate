#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Model Configuration ----------
DEBATER1_MODEL = "gpt-5-mini"  # First debater (OpenAI, smarter)
DEBATER2_MODEL = "gpt-5-nano"  # Second debater (OpenAI, cheaper/faster)
JUDGE_MODEL = "gpt-4o-mini"    # Judge (OpenAI)

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

# ---------- UI Configuration ----------
UI_CONFIG = {
    "title": "AI Debate System",
    "theme": "soft",
    "max_width": 1400,
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": False,
    "show_error": True
}

# ---------- Validation Limits ----------
VALIDATION_LIMITS = {
    "question_max_length": 1000,
    "max_question_length": 1000,  # Alias for main.py compatibility
    "max_tokens_min": 100,
    "min_tokens": 100,  # Alias for main.py compatibility
    "max_tokens_max": 4000,
    "max_tokens": 4000,  # Alias for main.py compatibility
    "debate_rounds_min": 1,
    "debate_rounds_max": 3,
    "rounds_min": 1,
    "min_rounds": 1,  # Alias for main.py compatibility
    "rounds_max": 3,
    "max_rounds": 3,  # Alias for main.py compatibility
    "min_models_required": 3,
    "min_models": 3  # Alias for main.py compatibility
}

# ---------- Cost Estimation Constants ----------
COST_ESTIMATION = {
    "words_per_token": 3,
    "tokens_per_million": 1_000_000,
    "average_cost_per_token": 0.25
}

# ---------- Default Values ----------
DEFAULT_VALUES = {
    "max_tokens": 1200,
    "debate_rounds": 1,
    "cost_limit": 0.10,
    "question_lines": 5,
    "question_max_lines": 10
}

# ---------- Preset Configurations ----------
PRESETS = {
    "fast": {
        "max_tokens": 800,
        "debate_rounds": 1,
        "cost_limit": 0.05,
        "description": "Quick responses with minimal cost"
    },
    "quality": {
        "max_tokens": 2000,
        "debate_rounds": 2,
        "cost_limit": 0.20,
        "description": "High-quality responses with multiple rounds"
    },
    "balanced": {
        "max_tokens": 1200,
        "debate_rounds": 1,
        "cost_limit": 0.10,
        "description": "Balanced approach between speed and quality"
    }
}
