"""Model backend implementations for HuggingFace and Ollama."""

import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict
from .core import set_seed
from typing import List, Dict, Any

def validate_token_budget(messages: List[Dict[str, str]], tokenizer, model_cfg) -> Dict[str, Any]:
    """Enhanced token budget validation with safety margins and AssertionError prevention"""
    # More accurate token counting using proper message formatting
    text = " ".join([msg.get("content", "") for msg in messages])
    input_tokens = len(tokenizer.encode(text))
    
    max_context = model_cfg.context_window
    max_output = model_cfg.max_new_tokens
    
    # Enhanced safety margins to prevent AssertionError
    # Conservative safety buffer: 150 tokens minimum (vs previous 50-80)
    safety_buffer = max(150, int(max_context * 0.1))  # At least 10% of context as buffer
    
    # Account for potential transformer internal overhead
    transformer_overhead = 50  # Additional buffer for internal processing
    total_overhead = safety_buffer + transformer_overhead
    
    # Check for max_position_embeddings bounds (common GPT model limit: 2048)
    position_limit = getattr(model_cfg, 'max_position_embeddings', 2048)
    effective_context = min(max_context, position_limit)
    
    # Validate against effective context window with overhead
    available_tokens = effective_context - total_overhead
    budget_ok = (input_tokens + max_output) <= available_tokens
    utilization = (input_tokens + max_output) / effective_context
    
    # Simple utilization check
    edge_case_risk = utilization > 0.5  # Conservative threshold
    risk_factors = []
    
    if edge_case_risk:
        risk_factors.append("high_utilization")
    
    return {
        "input_tokens": input_tokens,
        "max_output": max_output,
        "total_tokens": input_tokens + max_output,
        "context_window": max_context,
        "effective_context": effective_context,
        "available_tokens": available_tokens,
        "safety_buffer": total_overhead,
        "budget_ok": budget_ok,
        "utilization": utilization,
        "edge_case_risk": edge_case_risk,
        "risk_factors": risk_factors
    }

def calculate_max_output(input_tokens: int, model_cfg) -> int:
    """Calculate conservative max output tokens with simple, robust approach"""
    # Get effective context limit
    position_limit = getattr(model_cfg, 'max_position_embeddings', 2048)
    effective_context = min(model_cfg.context_window, position_limit)
    
    # Simple, conservative approach: use at most 50% of context window
    max_total_tokens = effective_context // 2
    
    # If input is already too large, use minimal output
    if input_tokens >= max_total_tokens:
        return 10  # Minimal viable output
    
    # Calculate available tokens for output with generous safety margin
    available_for_output = max_total_tokens - input_tokens
    
    # Respect user's desired max tokens but cap conservatively
    desired_max = getattr(model_cfg, 'max_new_tokens', 256)
    min_required = getattr(model_cfg, 'min_response_tokens', 10)
    
    # Choose the most conservative option
    recommended = min(desired_max, available_for_output, 512)  # Hard cap at 512 tokens
    
    # Ensure we meet minimum requirements
    return max(min_required, recommended)

def to_chat(messages: List[Dict[str, str]], tokenizer=None, use_harmony: bool = True, add_special_tokens: bool = True):
    """Format messages for model input with proper Harmony format support.
    
    For GPT-OSS models, this function supports both:
    1. Harmony format (use_harmony=True) - structured format with channels
    2. Standard chat template (use_harmony=False) - uses tokenizer.apply_chat_template()
    """
    
    if tokenizer and use_harmony:
        # Create spec-compliant Harmony format for structured responses
        text = create_harmony_chat_format(messages)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        return {"text": text, "enc": enc}
    elif tokenizer:
        # Use tokenizer's built-in template (recommended for GPT-OSS models)
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                # For GPT-OSS models, use the chat template with generation prompt
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback if template application fails
                text = simple_chat_format(messages)
        else:
            # Simple fallback formatting
            text = simple_chat_format(messages)
        
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        return {"text": text, "enc": enc}
    
    # No tokenizer available
    text = create_harmony_chat_format(messages) if use_harmony else simple_chat_format(messages)
    return {"text": text, "enc": None}

def create_harmony_chat_format(messages: List[Dict[str, str]]) -> str:
    """Create OpenAI Harmony spec-compliant chat format.
    
    Format: <|start|>{role}<|channel|>{channel}<|message|>{content}<|end|>
    
    Token IDs (for reference):
    - <|start|>: 200006
    - <|end|>: 200007  
    - <|return|>: 200002
    - <|message|>: 200008
    - <|channel|>: 200005
    
    Valid roles (hierarchy): system, developer, user, assistant, tool
    Valid channels: final, analysis, commentary
    """
    formatted_parts = []
    
    # Process all messages according to Harmony spec
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Map roles according to Harmony hierarchy
        if role == "system":
            # System messages use final channel for instructions
            formatted_parts.append(f"<|start|>system<|channel|>final<|message|>{content}<|end|>")
        elif role == "user":
            # User messages go to final channel
            formatted_parts.append(f"<|start|>user<|channel|>final<|message|>{content}<|end|>")
        elif role == "assistant":
            # Assistant messages should already be properly formatted, but ensure compliance
            if content.startswith("<|start|>"):
                # Already formatted, use as-is
                formatted_parts.append(content)
            else:
                # Plain text response, wrap in final channel
                formatted_parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")
        elif role == "developer":
            # Developer role (equivalent to system in chat template)
            formatted_parts.append(f"<|start|>developer<|channel|>final<|message|>{content}<|end|>")
        elif role == "tool":
            # Tool messages typically use commentary channel
            formatted_parts.append(f"<|start|>tool<|channel|>commentary<|message|>{content}<|end|>")
        else:
            # Unknown roles default to user with final channel
            formatted_parts.append(f"<|start|>user<|channel|>final<|message|>{content}<|end|>")
    
    return "".join(formatted_parts)

def simple_chat_format(messages: List[Dict[str, str]]) -> str:
    """Simple non-Harmony chat format for fallback."""
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"{role}: {content}\n"
    return text

class OllamaRunner:
    """Ollama model runner for text generation"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        # Map HF model names to Ollama model names
        self.ollama_model = self._get_ollama_model_name(cfg.model.model_name)
        
        print(f"Initializing Ollama backend...")
        print(f"HuggingFace model: {cfg.model.model_name}")
        print(f"Ollama model: {self.ollama_model}")
        
        # Verify Ollama is available
        if not self._check_ollama_available():
            raise RuntimeError("Ollama is not available or model is not installed")
        
        print(f"✅ Ollama backend ready with {self.ollama_model}")

    def _get_ollama_model_name(self, hf_model_name: str) -> str:
        """Map HuggingFace model names to Ollama equivalents"""
        mapping = {
            "openai/gpt-oss-20b": "gpt-oss:20b",
            "gpt-oss-20b": "gpt-oss:20b",
            "openai/gpt-oss-120b": "gpt-oss:120b",
            "gpt-oss-120b": "gpt-oss:120b"
        }
        return mapping.get(hf_model_name, "gpt-oss:20b")

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available and has the required model"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False
            return self.ollama_model.split(':')[0] in result.stdout
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
        except Exception as e:
            print(f"Warning: Unexpected error checking Ollama: {type(e).__name__}: {e}")
            return False

    def _format_messages_for_ollama(self, messages):
        """Format messages for Ollama chat interface"""
        conversation_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'system':
                conversation_parts.append(f"System: {content}")
            elif role == 'user':
                conversation_parts.append(f"User: {content}")
            elif role == 'assistant':
                conversation_parts.append(f"Assistant: {content}")
        conversation_parts.append("Assistant:")
        return "\n\n".join(conversation_parts)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Ollama"""
        formatted_prompt = self._format_messages_for_ollama(messages)
        
        try:
            result = subprocess.run([
                'ollama', 'run', self.ollama_model,
                formatted_prompt
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama generation failed: {result.stderr}")
            
            return result.stdout.strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")

    def generate_chat(self, messages):
        """Generate response for chat messages (detailed format)"""
        response_text = self.generate(messages)
        
        return {
            "prompt_text": self._format_messages_for_ollama(messages),
            "full_text": response_text,
            "gen_text": response_text,
            "model": asdict(self.cfg.model),
            "gen_params": {
                "temperature": self.cfg.model.temperature,
                "top_p": self.cfg.model.top_p,
                "seed": self.cfg.model.seed,
                "backend": "ollama"
            }
        }

class HuggingFaceRunner:
    """HuggingFace model runner for text generation"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        print(f"Loading tokenizer: {cfg.model.model_name}")
        self.tok = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=True)
        
        # Ensure pad token is set to prevent issues
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        print(f"Loading model with dtype: {cfg.model.dtype}")
        dtype = getattr(torch, cfg.model.dtype) if hasattr(torch, cfg.model.dtype) else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name, 
            torch_dtype=dtype, 
            device_map="auto"
        )
        self.model.eval()
        
        # Store model capabilities for enhanced validation
        self.max_position_embeddings = getattr(self.model.config, 'max_position_embeddings', 2048)
        if hasattr(cfg.model, 'max_position_embeddings'):
            # User override
            self.max_position_embeddings = min(self.max_position_embeddings, cfg.model.max_position_embeddings)
        
        # Update config with detected limits
        cfg.model.max_position_embeddings = self.max_position_embeddings
        
        print(f"Model loaded on device: {self.model.device}")
        print(f"Max position embeddings: {self.max_position_embeddings}")
        print(f"Context window: {cfg.model.context_window}")

    def _truncate_messages(self, messages: List[Dict[str, str]], target_tokens: int) -> List[Dict[str, str]]:
        """Enhanced message truncation with conversation-aware and edge-case handling"""
        if not messages:
            return messages
        
        # More aggressive truncation for edge case prevention
        # Reserve 20% of target for safety margin
        effective_target = int(target_tokens * 0.8)
        
        result = []
        total_tokens = 0
        
        # Always prioritize the last message (current user request)
        last_msg = messages[-1]
        last_content = last_msg.get("content", "")
        last_tokens = len(self.tok.encode(last_content))
        
        if last_tokens < effective_target:
            result.append(last_msg)
            total_tokens += last_tokens
        else:
            # Last message is too long - more aggressive truncation
            words = last_content.split()
            # Use only 30% of target for last message content to leave room for context
            max_words_for_last = max(10, effective_target // 6)  # Very conservative
            truncated_words = words[:max_words_for_last]
            
            truncated_content = " ".join(truncated_words)
            if len(truncated_words) < len(words):
                truncated_content += "... [truncated for safety]"
            
            result.append({
                "role": last_msg.get("role", "user"),
                "content": truncated_content
            })
            
            # Update token count with actual truncated content
            total_tokens = len(self.tok.encode(truncated_content))
            return result
        
        # Handle system message with conservative approach
        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            sys_content = system_msg.get("content", "")
            sys_tokens = len(self.tok.encode(sys_content))
            
            # Only include system message if it's reasonable size
            if sys_tokens < effective_target * 0.3 and total_tokens + sys_tokens < effective_target:
                result.insert(0, system_msg)
                total_tokens += sys_tokens
            elif sys_tokens >= effective_target * 0.3:
                # System message too long, truncate it
                sys_words = sys_content.split()
                max_sys_words = max(5, effective_target // 15)  # Very conservative for system
                truncated_sys = " ".join(sys_words[:max_sys_words])
                if len(sys_words) > max_sys_words:
                    truncated_sys += "... [truncated]"
                
                truncated_sys_msg = {
                    "role": "system",
                    "content": truncated_sys
                }
                truncated_sys_tokens = len(self.tok.encode(truncated_sys))
                
                if total_tokens + truncated_sys_tokens < effective_target:
                    result.insert(0, truncated_sys_msg)
                    total_tokens += truncated_sys_tokens
        
        # Add conversation history with more conservative approach
        history_messages = []
        start_idx = 1 if system_msg else 0
        end_idx = len(messages) - 1  # Exclude last message (already added)
        
        if start_idx < end_idx:
            history_messages = messages[start_idx:end_idx]
        
        # Add history in reverse order (most recent first) with aggressive limits
        remaining_budget = effective_target - total_tokens
        
        for msg in reversed(history_messages):
            msg_content = msg.get("content", "")
            msg_tokens = len(self.tok.encode(msg_content))
            
            # Very conservative: only add if message is small and we have plenty of budget
            if msg_tokens < remaining_budget * 0.4 and total_tokens + msg_tokens < effective_target:
                # Insert conversation history in correct order
                insert_pos = len(result) - 1 if len(result) > 0 else 0
                result.insert(insert_pos, msg)
                total_tokens += msg_tokens
                remaining_budget = effective_target - total_tokens
            else:
                # Stop adding history if budget is tight
                break
        
        # Final validation and emergency truncation
        final_tokens = sum(len(self.tok.encode(m.get("content", ""))) for m in result)
        if final_tokens > effective_target:
            # Emergency: keep only last message and minimal system message
            emergency_result = [result[-1]]  # Last message (user request)
            
            # Add minimal system message if exists
            if result and result[0].get("role") == "system":
                sys_content = result[0].get("content", "")
                sys_words = sys_content.split()[:10]  # Only first 10 words
                emergency_sys = {
                    "role": "system", 
                    "content": " ".join(sys_words) + ("..." if len(sys_words) == 10 else "")
                }
                emergency_result.insert(0, emergency_sys)
            
            return emergency_result
        
        return result

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Simple generation interface"""
        # Input validation
        if isinstance(messages, str):
            raise TypeError(f"Expected List[Dict[str, str]] but got str: '{messages}'. Use format: [{{'role': 'user', 'content': 'your message'}}]")
        if not isinstance(messages, list):
            raise TypeError(f"Expected List[Dict[str, str]] but got {type(messages)}")
        if not messages:
            raise ValueError("Messages list cannot be empty")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"Message {i} must be a dict, got {type(msg)}")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing required 'content' field")
        
        result = self.generate_chat(messages)
        return result["gen_text"]
        
    def _inject_system_prompt(self, messages):
        """Inject configured system prompt if enabled"""
        if not hasattr(self.cfg, 'system_prompt') or not self.cfg.system_prompt.enabled:
            return messages
            
        # Check if system message already exists
        has_system = any(msg.get('role') == 'system' for msg in messages)
        
        if not has_system:
            # Add system prompt as first message
            system_content = self.cfg.system_prompt.render_template()
            system_message = {"role": "system", "content": system_content}
            return [system_message] + messages
        else:
            # System message already exists, don't modify
            return messages

    def generate_chat(self, messages):
        """Generate response for chat messages with dynamic token adjustment"""
        # Input validation (same as generate method)
        if isinstance(messages, str):
            raise TypeError(f"Expected List[Dict[str, str]] but got str: '{messages}'. Use format: [{{'role': 'user', 'content': 'your message'}}]")
        if not isinstance(messages, list):
            raise TypeError(f"Expected List[Dict[str, str]] but got {type(messages)}")
        if not messages:
            raise ValueError("Messages list cannot be empty")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"Message {i} must be a dict, got {type(msg)}")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing required 'content' field")
        
        cfg = self.cfg
        set_seed(cfg.model.seed)
        
        # Inject system prompt if enabled and not already present
        messages = self._inject_system_prompt(messages)
        
        # Validate token budget and calculate dynamic allocation
        budget = validate_token_budget(messages, self.tok, cfg.model)
        
        # Enhanced input validation with edge case and AssertionError prevention
        original_tokens = budget["input_tokens"]
        needs_truncation = False
        
        # Hard upper limit to prevent AssertionError using conservative approach
        # Uses either 500 tokens or 50% of context window, whichever is smaller
        hard_token_limit = min(500, int(budget["effective_context"] * 0.5))  # Very conservative: max 500 tokens
        
        # Check multiple risk factors
        if budget["input_tokens"] > hard_token_limit:  # Hard limit to prevent AssertionError
            needs_truncation = True
            print(f"Hard token limit triggered: {budget['input_tokens']} > {hard_token_limit}")
        elif budget["input_tokens"] > budget["effective_context"] * 0.8:  # More than 80% of context
            needs_truncation = True
        elif budget.get("edge_case_risk", False):  # Risk factors detected
            needs_truncation = True
        elif not budget["budget_ok"]:  # Budget validation failed
            needs_truncation = True
        
        if needs_truncation:
            # More aggressive truncation to prevent AssertionError
            # Use the hard limit or 60% of context, whichever is smaller
            target_tokens = min(hard_token_limit, int(budget["effective_context"] * 0.6))
            truncated_messages = self._truncate_messages(messages, target_tokens)
            budget = validate_token_budget(truncated_messages, self.tok, cfg.model)
            messages = truncated_messages
            
            debug_tokens = getattr(cfg.model, 'debug_token_budget', False)
            if debug_tokens or original_tokens != budget["input_tokens"]:
                print(f"Warning: Input truncated from {original_tokens} to {budget['input_tokens']} tokens (risk factors: {budget.get('risk_factors', [])})")
        
        dynamic_max_tokens = calculate_max_output(budget["input_tokens"], cfg.model)
        
        # Additional validation after calculation
        total_projected = budget["input_tokens"] + dynamic_max_tokens
        if total_projected > budget["effective_context"] - budget["safety_buffer"]:
            # Reduce output tokens to stay within safe limits
            max_safe_output = budget["effective_context"] - budget["input_tokens"] - budget["safety_buffer"]
            dynamic_max_tokens = max(1, max_safe_output)
            print(f"Warning: Reduced output tokens to {dynamic_max_tokens} to prevent assertion errors")
        
        # Enhanced memory management with edge case consideration
        needs_cleanup = any([
            budget["input_tokens"] > 1200,  # Reduced threshold
            dynamic_max_tokens > 400,      # Reduced threshold  
            budget.get("edge_case_risk", False),  # Risk factors detected
            budget["utilization"] > 0.85    # High utilization
        ])
        
        if needs_cleanup:
            debug_tokens = getattr(cfg.model, 'debug_token_budget', False)
            if debug_tokens:
                cleanup_reasons = []
                if budget["input_tokens"] > 1200: cleanup_reasons.append("large_input")
                if dynamic_max_tokens > 400: cleanup_reasons.append("large_output")
                if budget.get("edge_case_risk", False): cleanup_reasons.append("edge_case_risk")
                if budget["utilization"] > 0.85: cleanup_reasons.append("high_utilization")
                print(f"Proactive cleanup (reasons: {cleanup_reasons}): input={budget['input_tokens']}, output={dynamic_max_tokens}")
            cleanup_model_memory()
        
        # Enhanced budget status reporting
        if not budget["budget_ok"]:
            risk_info = f" (risks: {budget.get('risk_factors', [])})" if budget.get('risk_factors') else ""
            print(f"Warning: Token budget exceeded - input: {budget['input_tokens']}, output: {dynamic_max_tokens}, utilization: {budget['utilization']:.1%}{risk_info}")
        
        # Enhanced edge case handling with better diagnostics
        min_required = getattr(cfg.model, 'min_response_tokens', 10)
        
        if dynamic_max_tokens < 5:
            # Absolute minimum for any meaningful output
            dynamic_max_tokens = 5
            print(f"Warning: Input extremely long ({budget['input_tokens']} tokens), forced to minimal output ({dynamic_max_tokens} tokens)")
        elif dynamic_max_tokens < min_required:
            # Try to honor minimum, but be conservative
            safe_minimum = min(min_required, budget["available_tokens"] // 4)
            if safe_minimum > dynamic_max_tokens:
                dynamic_max_tokens = safe_minimum
                print(f"Warning: Increased to safe minimum {dynamic_max_tokens} tokens (was {safe_minimum})")
            else:
                print(f"Warning: Cannot meet minimum {min_required} tokens due to constraints, using {dynamic_max_tokens} tokens")
        
        # Log edge case handling for debugging
        if budget.get("edge_case_risk", False):
            debug_info = {
                "input_tokens": budget["input_tokens"],
                "output_tokens": dynamic_max_tokens,
                "risk_factors": budget.get("risk_factors", []),
                "utilization": budget["utilization"]
            }
            print(f"Edge case handling: {debug_info}")
        
        # Enhanced debug information
        debug_tokens = getattr(cfg.model, 'debug_token_budget', False)
        if debug_tokens:
            debug_info = {
                "input": budget['input_tokens'],
                "output": dynamic_max_tokens,
                "total": budget['input_tokens'] + dynamic_max_tokens,
                "effective_context": budget['effective_context'],
                "utilization": f"{budget['utilization']:.1%}",
                "safety_buffer": budget['safety_buffer']
            }
            if budget.get('edge_case_risk'):
                debug_info["risk_factors"] = budget['risk_factors']
            print(f"Token budget analysis: {debug_info}")
        
        # Format messages
        try:
            packed = to_chat(
                messages, 
                self.tok, 
                use_harmony=cfg.model.use_harmony_chat_template, 
                add_special_tokens=cfg.model.add_special_tokens
            )
        except Exception as e:
            raise RuntimeError(f"Failed to format messages: {e}")
        
        # Move to model device
        try:
            inputs = {k: v.to(self.model.device) for k, v in packed["enc"].items()}
        except Exception as e:
            raise RuntimeError(f"Failed to move inputs to device {self.model.device}: {e}")
        
        # Enhanced generation with AssertionError handling and fallback
        generation_attempts = 0
        max_attempts = 3
        out = None
        
        while generation_attempts < max_attempts and out is None:
            generation_attempts += 1
            current_max_tokens = dynamic_max_tokens
            
            # Progressive token reduction on retry attempts
            if generation_attempts > 1:
                reduction_factor = 0.7 ** (generation_attempts - 1)  # 70%, 49% of original
                current_max_tokens = max(5, int(dynamic_max_tokens * reduction_factor))
                print(f"Attempt {generation_attempts}: Reducing output tokens to {current_max_tokens}")
            
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=current_max_tokens,
                        temperature=cfg.model.temperature,
                        top_p=cfg.model.top_p,
                        do_sample=cfg.model.temperature > 0,
                        pad_token_id=self.tok.eos_token_id,  # Ensure proper padding
                    )
                    
                # If successful, update dynamic_max_tokens for reporting
                if out is not None:
                    dynamic_max_tokens = current_max_tokens
                    break
                    
            except AssertionError as e:
                # Specific handling for AssertionError - the main issue
                error_msg = str(e)
                print(f"AssertionError on attempt {generation_attempts}: {error_msg}")
                
                if generation_attempts >= max_attempts:
                    # Final attempt failed, provide detailed error
                    cleanup_model_memory()
                    raise RuntimeError(f"AssertionError during generation after {max_attempts} attempts (input: {budget['input_tokens']} tokens, final output attempt: {current_max_tokens} tokens). This suggests token limit exceeded model's internal constraints. Original error: {e}")
                
                # Continue to next attempt with reduced tokens
                cleanup_model_memory()  # Clean memory between attempts
                continue
                
            except torch.cuda.OutOfMemoryError as e:
                cleanup_model_memory()
                if generation_attempts >= max_attempts:
                    raise RuntimeError(f"GPU out of memory during generation (input: {budget['input_tokens']} tokens, output: {current_max_tokens} tokens). Try reducing input length or max_new_tokens. Original error: {e}")
                continue
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    cleanup_model_memory()
                    if generation_attempts >= max_attempts:
                        raise RuntimeError(f"CUDA/Memory error during generation (input: {budget['input_tokens']} tokens, output: {current_max_tokens} tokens): {e}")
                    continue
                else:
                    raise RuntimeError(f"Model generation failed (input: {budget['input_tokens']} tokens, output: {current_max_tokens} tokens): {e}")
                    
            except Exception as e:
                error_type = type(e).__name__
                if generation_attempts >= max_attempts:
                    raise RuntimeError(f"Unexpected error during generation after {max_attempts} attempts (input: {budget['input_tokens']} tokens, output: {current_max_tokens} tokens, type: {error_type}): {e}")
                
                # For other exceptions, still try to continue if we have attempts left
                print(f"Unexpected {error_type} on attempt {generation_attempts}: {e}")
                cleanup_model_memory()
                continue
        
        # Ensure we have a valid result
        if out is None:
            raise RuntimeError(f"Failed to generate after {max_attempts} attempts with progressive token reduction")
        
        # Decode outputs
        try:
            full = self.tok.decode(out[0], skip_special_tokens=False)
            gen_only = self.tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        except Exception as e:
            raise RuntimeError(f"Failed to decode outputs: {e}")
        
        # Basic response analysis
        response_length = len(gen_only.split())
        
        # Assess response quality with robust error handling
        try:
            from .judges import assess_response_quality
            quality = assess_response_quality(gen_only)
            quality_info = {
                "completeness_score": quality["completeness_score"],
                "export_ready": quality["export_ready"],
                "issues": quality["issues"]
            }
        except ImportError:
            # Judges module not available, use basic assessment
            quality_info = {
                "completeness_score": 1.0 if len(gen_only.strip()) > 10 else 0.5,
                "export_ready": len(gen_only.strip()) > 0,
                "issues": ["quality_assessment_unavailable"]
            }
        except Exception as e:
            # Quality assessment failed, provide fallback
            quality_info = {
                "completeness_score": 0.8,
                "export_ready": len(gen_only.strip()) > 0,
                "issues": [f"quality_assessment_error: {type(e).__name__}"]
            }
        
        return {
            "prompt_text": packed["text"],
            "full_text": full,
            "gen_text": gen_only,
            "input_ids": inputs["input_ids"].tolist(),
            "generated_ids": out[0].tolist(),
            "model": asdict(cfg.model),
            "gen_params": {
                "max_new_tokens": dynamic_max_tokens,  # Record actual value used
                "configured_max_tokens": cfg.model.max_new_tokens,  # Record configured value
                "temperature": cfg.model.temperature,
                "top_p": cfg.model.top_p,
                "seed": cfg.model.seed,
                "backend": "huggingface"
            },
            "token_info": {
                "input_tokens": budget["input_tokens"],
                "output_tokens": dynamic_max_tokens,
                "response_length_words": response_length,
                "budget_utilization": budget["utilization"],
                "dynamic_adjustment": dynamic_max_tokens != cfg.model.max_new_tokens,
                "effective_context": budget["effective_context"],
                "safety_buffer": budget["safety_buffer"],
                "edge_case_risk": budget.get("edge_case_risk", False),
                "risk_factors": budget.get("risk_factors", []),
                "generation_attempts": generation_attempts
            },
            "quality_info": quality_info
        }

# Compatibility aliases
HFRunner = HuggingFaceRunner

def cleanup_model_memory():
    """Enhanced memory cleanup for stability and AssertionError prevention"""
    try:
        import gc
        import torch
        
        # More aggressive garbage collection
        for _ in range(3):  # Multiple GC passes
            gc.collect()
        
        # Enhanced CUDA memory management
        if torch.cuda.is_available():
            # Get memory info before cleanup
            allocated_before = torch.cuda.memory_allocated()
            
            # Multiple cache clear attempts
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Second pass
            
            # Force memory cleanup
            if hasattr(torch.cuda, 'memory_stats'):
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            
            # Final synchronization
            torch.cuda.synchronize()
            
            allocated_after = torch.cuda.memory_allocated()
            freed_mb = (allocated_before - allocated_after) / 1024 / 1024
            
            if freed_mb > 10:  # Only report significant memory freeing
                print(f"Memory cleanup freed {freed_mb:.1f} MB")
            
        return True
    except Exception as e:
        print(f"Warning: Memory cleanup partially failed: {e}")
        return False

def create_runner(cfg):
    """Create appropriate runner based on backend configuration"""
    if cfg.model.backend == "ollama":
        return OllamaRunner(cfg)
    elif cfg.model.backend == "huggingface":
        return HuggingFaceRunner(cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.model.backend}")