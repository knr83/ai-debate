#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr
from typing import Dict, Any, List, Tuple

from config import AVAILABLE_MODELS, DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL, PRESETS, VALIDATION_LIMITS, DEFAULT_VALUES, UI_CONFIG


class DebateUI:
    """UI class for AI debate system"""
    
    def __init__(self):
        self._css = None
    
    def _get_custom_css(self) -> str:
        """Returns custom CSS styles from file"""
        if self._css is None:
            try:
                with open('styles.css', 'r', encoding='utf-8') as f:
                    css_content = f.read()
                    # Replace placeholder with actual max width value
                    css_content = css_content.replace('1400px', f'{UI_CONFIG["max_width"]}px')
                    self._css = css_content
            except FileNotFoundError:
                raise FileNotFoundError("CSS file 'styles.css' not found")
        return self._css
    
    def create_interface(self, debate_function: callable, cost_estimation_function: callable) -> gr.Blocks:
        """Creates and configures Gradio interface"""
        
        with gr.Blocks(css=self._get_custom_css(), title=UI_CONFIG["title"], theme=gr.themes.Soft()) as demo:
            self._create_header()
            self._create_advanced_settings()
            self._create_main_interface()
            self._create_output_section()
            self._create_examples()
            
            # Interface initialization
            demo.load(
                fn=self._update_settings_display,
                inputs=[
                    self.max_tokens_slider, 
                    self.debate_rounds, 
                    self.cost_limit, 
                    self.model_selector
                ],
                outputs=[self.settings_display]
            )
            
            # Setup event handlers inside Blocks context
            self._setup_ui_handlers()
            self._setup_preset_handlers()
            self._setup_settings_handlers()
            self._setup_function_handlers(debate_function, cost_estimation_function)
        
        return demo
    
    def _create_header(self):
        """Creates interface header"""
        gr.HTML("<h1 class='main-header'>🤖 AI Debate System</h1>")
        gr.HTML(
            "<p class='header-description'>"
            "Get AI-powered answers through intelligent debate and judgment</p>"
        )
    
    def _create_advanced_settings(self):
        """Creates advanced settings section"""
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            self._create_settings_rows()
            self._create_preset_buttons()
    
    def _create_settings_rows(self):
        """Creates settings rows with parameters and controls"""
        with gr.Row():
            self._create_model_parameters()
            self._create_model_selection()
        
        with gr.Row():
            self._create_cost_control()
            self._create_quick_presets()
    
    def _create_model_parameters(self):
        """Creates model parameter settings"""
        with gr.Column():
            gr.HTML("<h4>Model Parameters:</h4>")
            
            # Max tokens setting
            self.max_tokens_slider = self._create_slider(
                "max_tokens", 
                "Max Output Tokens", 
                "Maximum tokens for each model response",
                step=100
            )
            
            # Debate rounds setting
            self.debate_rounds = self._create_slider(
                "debate_rounds", 
                "Debate Rounds", 
                "Number of debate rounds (1-3)",
                step=1
            )
    
    def _create_slider(self, param_name: str, label: str, info: str, step: int = 1) -> gr.Slider:
        """Factory method for creating sliders"""
        limits_key = f"{param_name}_min"
        max_key = f"{param_name}_max"
        
        return gr.Slider(
            minimum=VALIDATION_LIMITS[limits_key],
            maximum=VALIDATION_LIMITS[max_key],
            value=DEFAULT_VALUES[param_name],
            step=step,
            label=label,
            info=info
        )
    
    def _create_model_selection(self):
        """Creates model selector"""
        with gr.Column():
            gr.HTML("<h4>Model Selection:</h4>")
            
            # Model selection
            self.model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=[DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL],
                label="Select Models (Debater1, Debater2, Judge)",
                multiselect=True,
                info="Choose which models to use"
            )
    
    def _create_cost_control(self):
        """Creates cost control elements"""
        with gr.Column():
            gr.HTML("<h4>Cost Control:</h4>")
            
            # Cost limit
            self.cost_limit = gr.Number(
                value=DEFAULT_VALUES["cost_limit"], precision=2,
                label="Cost Limit ($)",
                info="Stop if total cost exceeds this amount"
            )
            
            # Cost estimation button
            self.estimate_btn = gr.Button("💰 Estimate Cost", variant="secondary")
            self.cost_estimate = gr.Textbox(
                label="Estimated Cost",
                interactive=False
            )
    
    def _create_quick_presets(self):
        """Creates quick presets section"""
        with gr.Column():
            gr.HTML("<h4>Quick Presets:</h4>")
    
    def _create_preset_buttons(self):
        """Creates preset buttons"""
        with gr.Row():
            self.fast_btn = gr.Button("🚀 Fast", variant="secondary", size="sm", elem_classes=["preset-btn"])
            self.quality_btn = gr.Button("⭐ Quality", variant="secondary", size="sm", elem_classes=["preset-btn"])
            self.balanced_btn = gr.Button("⚖️ Balanced", variant="secondary", size="sm", elem_classes=["preset-btn"])
    
    def _create_main_interface(self):
        """Creates main interface"""
        with gr.Row():
            with gr.Column(scale=2):
                self._create_input_section()
            
            with gr.Column(scale=1):
                self._create_settings_display()
    
    def _create_input_section(self):
        """Creates input section"""
        gr.HTML("<h4>Enter your question:</h4>")
        
        self.question_input = gr.Textbox(
            label="",
            placeholder=f"Type your question here... (max {VALIDATION_LIMITS['question_max_length']} characters)",
            lines=DEFAULT_VALUES["question_lines"],
            max_lines=DEFAULT_VALUES["question_max_lines"],
            max_length=VALIDATION_LIMITS["question_max_length"],
            interactive=True,
            show_copy_button=True
        )
        
        # Character counter
        self.char_counter = gr.HTML(
            value=f"<div class='char-counter'>Characters: 0/{VALIDATION_LIMITS['question_max_length']}</div>",
            label=""
        )
        
        with gr.Row():
            self.submit_btn = gr.Button("Get AI Debate Answer", variant="primary", size="lg")
            self.clear_btn = gr.Button("Clear All", variant="secondary", size="lg")
    
    def _create_settings_display(self):
        """Creates settings display"""
        with gr.Accordion("Active Configuration", open=False):
            self.settings_display = gr.JSON(value={})
    
    def _create_output_section(self):
        """Creates output section"""
        with gr.Row():
            with gr.Column():
                self._create_final_answer_section()
            
            with gr.Column():
                self._create_meta_output_section()
        
        self.cost_output = gr.Textbox(
            label="Cost Information",
            interactive=False
        )
    
    def _create_final_answer_section(self):
        """Creates final answer section"""
        gr.HTML("<h4>Final Answer</h4>")
        
        self.final_answer_output = gr.Textbox(
            label="",
            lines=8,
            max_lines=15,
            interactive=False,
            show_copy_button=True
        )
    
    def _create_meta_output_section(self):
        """Creates meta output section"""
        self.meta_output = gr.Textbox(
            label="Debate Details & Individual AI Responses",
            lines=20,
            max_lines=50,
            interactive=False,
            show_copy_button=True
        )
    
    def _create_examples(self):
        """Creates example questions"""
        gr.Examples(
            examples=[
                "What are the main differences between Python and JavaScript?",
                "Explain the concept of machine learning in simple terms",
                "What are the pros and cons of remote work?",
                "How does photosynthesis work?"
            ],
            inputs=self.question_input
        )
    
    def setup_event_handlers(self, debate_function: callable, cost_estimation_function: callable):
        """Sets up event handlers"""
        self._setup_ui_handlers()
        self._setup_preset_handlers()
        self._setup_settings_handlers()
        self._setup_function_handlers(debate_function, cost_estimation_function)
    
    def _setup_ui_handlers(self):
        """Sets up UI-related event handlers"""
        # Character counter update
        def update_char_counter(text):
            return f"<div class='char-counter'>Characters: {len(text)}/{VALIDATION_LIMITS['question_max_length']}</div>"
        
        self.question_input.change(
            fn=update_char_counter, 
            inputs=[self.question_input], 
            outputs=[self.char_counter]
        )
        
        # Clear all fields
        def clear_all():
            return "", "", "", "", {}
        
        self.clear_btn.click(
            fn=clear_all,
            outputs=[
                self.question_input, 
                self.final_answer_output, 
                self.cost_output, 
                self.char_counter, 
                self.settings_display
            ]
        )
        

        

    
    def _setup_preset_handlers(self):
        """Sets up preset button handlers"""
        # Universal preset function
        def apply_preset(preset_name: str):
            preset = PRESETS[preset_name]
            return preset["max_tokens"], preset["debate_rounds"], preset["cost_limit"], [DEBATER1_MODEL, DEBATER2_MODEL, JUDGE_MODEL]
        
        # Button to preset mapping
        preset_buttons = {
            self.fast_btn: "fast",
            self.quality_btn: "quality", 
            self.balanced_btn: "balanced"
        }
        
        # Apply presets using mapping
        outputs = [self.max_tokens_slider, self.debate_rounds, self.cost_limit, self.model_selector]
        for button, preset_name in preset_buttons.items():
            button.click(
                fn=lambda p=preset_name: apply_preset(p),
                outputs=outputs
            )
    
    def _setup_settings_handlers(self):
        """Sets up settings change handlers"""
        # Update settings display when parameters change
        def update_settings_wrapper(*args):
            return self._update_settings_display(*args)
        
        # Map components to their change handlers
        components_to_watch = [
            self.max_tokens_slider, 
            self.debate_rounds, 
            self.cost_limit, 
            self.model_selector
        ]
        
        for component in components_to_watch:
            component.change(
                fn=update_settings_wrapper,
                inputs=components_to_watch,
                outputs=self.settings_display
            )
    
    def _setup_function_handlers(self, debate_function: callable, cost_estimation_function: callable):
        """Sets up main function handlers"""
        # Cost estimation button
        self.estimate_btn.click(
            fn=cost_estimation_function,
            inputs=[self.question_input, self.max_tokens_slider, self.debate_rounds, self.model_selector],
            outputs=[self.cost_estimate]
        )
        
        # Main submit button
        self.submit_btn.click(
            fn=debate_function,
            inputs=[
                self.question_input, 
                self.max_tokens_slider, 
                self.debate_rounds,
                self.cost_limit, 
                self.model_selector
            ],
            outputs=[self.final_answer_output, self.meta_output, self.cost_output]
        )
    
    def _update_settings_display(self, max_tokens, rounds, cost_lim, models) -> Dict[str, Any]:
        """Updates settings display"""
        return {
            "max_tokens": max_tokens,
            "debate_rounds": rounds,
            "cost_limit": cost_lim,
            "selected_models": models
        }
