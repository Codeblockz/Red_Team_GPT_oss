"""Model backend implementations for HuggingFace and Ollama."""

import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict
from .core import set_seed
from typing import List, Dict, Any

def validate_token_budget(messages: List[Dict[str, str]], tokenizer, model_cfg) -> Dict[str, Any]:
    """Simplified token budget validation using standard practices"""
    # More accurate token counting using proper message formatting
    text = " ".join([msg.get("content", "") for msg in messages])
    input_tokens = len(tokenizer.encode(text))
    
    max_context = model_cfg.context_window
    max_output = model_cfg.max_new_tokens
    
    # Standard safety buffer: 10% of context window
    safety_buffer = int(max_context * 0.1)
    
    # Check for max_position_embeddings bounds
    position_limit = getattr(model_cfg, 'max_position_embeddings', 2048)
    effective_context = min(max_context, position_limit)
    
    # Validate against effective context window with standard buffer
    available_tokens = effective_context - safety_buffer
    budget_ok = (input_tokens + max_output) <= available_tokens
    utilization = (input_tokens + max_output) / effective_context
    
    # Standard utilization check (80% is reasonable for most models)
    edge_case_risk = utilization > 0.8
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
        "safety_buffer": safety_buffer,
        "budget_ok": budget_ok,
        "utilization": utilization,
        "edge_case_risk": edge_case_risk,
        "risk_factors": risk_factors
    }

def calculate_max_output(input_tokens: int, model_cfg) -> int:
    """Calculate reasonable max output tokens using standard practices"""
    # Get effective context limit
    position_limit = getattr(model_cfg, 'max_position_embeddings', 2048)
    effective_context = min(model_cfg.context_window, position_limit)
    
    # Standard approach: use up to 80% of context window total
    safety_buffer = int(effective_context * 0.1)
    max_total_tokens = effective_context - safety_buffer
    
    # Calculate available tokens for output
    available_for_output = max_total_tokens - input_tokens
    
    # If input is very large, still allow reasonable output
    if available_for_output < 50:
        return max(20, available_for_output)  # Minimum viable output
    
    # Respect user's desired max tokens
    desired_max = getattr(model_cfg, 'max_new_tokens', 256)
    min_required = getattr(model_cfg, 'min_response_tokens', 50)  # Increased from 10
    
    # Use desired max unless it exceeds available space
    recommended = min(desired_max, available_for_output)
    
    # Ensure reasonable minimum for quality output
    return max(min_required, recommended)

def to_chat(messages: List[Dict[str, str]], tokenizer=None, use_harmony: bool = False, add_special_tokens: bool = True):
    """Format messages for model input using tokenizer's built-in chat template.
    
    For GPT-OSS models, this function prioritizes:
    1. Standard chat template (use_harmony=False) - uses tokenizer.apply_chat_template() (RECOMMENDED)
    2. Harmony format (use_harmony=True) - structured format with channels (LEGACY)
    """
    
    if tokenizer:
        if not use_harmony and hasattr(tokenizer, 'apply_chat_template'):
            try:
                # Use tokenizer's built-in template (recommended approach from simple_harmony.ipynb)
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
                return {"text": text, "enc": enc}
            except Exception:
                # Fallback if template application fails
                pass
        
        if use_harmony:
            # Create spec-compliant Harmony format for structured responses
            text = create_harmony_chat_format(messages)
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
            return {"text": text, "enc": enc}
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
        """Simple message truncation that preserves message integrity"""
        if not messages:
            return messages
        
        # Keep recent messages that fit within target
        result = []
        total_tokens = 0
        
        # Always include the most recent message (user request)
        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = len(self.tok.encode(content))
            
            if total_tokens + tokens <= target_tokens:
                result.insert(0, msg)
                total_tokens += tokens
            elif msg.get("role") == "system" and not any(m.get("role") == "system" for m in result):
                # Always try to include system message, truncate if needed
                words = content.split()
                max_words = min(100, len(words))  # Reasonable system message length
                truncated_content = " ".join(words[:max_words])
                if max_words < len(words):
                    truncated_content += "..."
                
                truncated_msg = {"role": "system", "content": truncated_content}
                truncated_tokens = len(self.tok.encode(truncated_content))
                
                if total_tokens + truncated_tokens <= target_tokens:
                    result.insert(0, truncated_msg)
                    total_tokens += truncated_tokens
            else:
                # Skip this message to stay within budget
                continue
        
        return result if result else [messages[-1]]  # Always return at least the last message

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
        """Generate response for chat messages using simplified, reliable approach"""
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
        
        cfg = self.cfg
        set_seed(cfg.model.seed)
        
        # Inject system prompt if enabled and not already present
        messages = self._inject_system_prompt(messages)
        
        # Simple token budget validation
        budget = validate_token_budget(messages, self.tok, cfg.model)
        
        # Truncate if needed (simple approach)
        if not budget["budget_ok"] or budget.get("edge_case_risk", False):
            target_tokens = int(budget["effective_context"] * 0.7)  # Use 70% for input
            messages = self._truncate_messages(messages, target_tokens)
            budget = validate_token_budget(messages, self.tok, cfg.model)
        
        # Calculate output tokens (simple approach)
        max_output_tokens = calculate_max_output(budget["input_tokens"], cfg.model)
        
        # Format messages using the proven approach from simple_harmony.ipynb
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
        
        # Simple, reliable generation (based on simple_harmony.ipynb pattern)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    temperature=cfg.model.temperature,
                    top_p=cfg.model.top_p,
                    do_sample=cfg.model.temperature > 0,
                    pad_token_id=self.tok.eos_token_id,
                )
        except Exception as e:
            # Simple fallback: reduce tokens and try once more
            reduced_tokens = max(20, max_output_tokens // 2)
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=reduced_tokens,
                        temperature=cfg.model.temperature,
                        top_p=cfg.model.top_p,
                        do_sample=cfg.model.temperature > 0,
                        pad_token_id=self.tok.eos_token_id,
                    )
                max_output_tokens = reduced_tokens
                print(f"Warning: Reduced output tokens to {reduced_tokens} due to generation error")
            except Exception as e2:
                raise RuntimeError(f"Model generation failed: {e2}")
        
        # Decode outputs
        try:
            full = self.tok.decode(out[0], skip_special_tokens=False)
            gen_only = self.tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        except Exception as e:
            raise RuntimeError(f"Failed to decode outputs: {e}")
        
        # Simple response quality assessment
        response_length = len(gen_only.split())
        quality_info = {
            "completeness_score": 1.0 if len(gen_only.strip()) > 20 else 0.5,
            "export_ready": len(gen_only.strip()) > 0,
            "issues": []
        }
        
        return {
            "prompt_text": packed["text"],
            "full_text": full,
            "gen_text": gen_only,
            "input_ids": inputs["input_ids"].tolist(),
            "generated_ids": out[0].tolist(),
            "model": asdict(cfg.model),
            "gen_params": {
                "max_new_tokens": max_output_tokens,
                "temperature": cfg.model.temperature,
                "top_p": cfg.model.top_p,
                "seed": cfg.model.seed,
                "backend": "huggingface"
            },
            "token_info": {
                "input_tokens": budget["input_tokens"],
                "output_tokens": max_output_tokens,
                "response_length_words": response_length,
                "budget_utilization": budget["utilization"],
                "effective_context": budget["effective_context"],
                "safety_buffer": budget["safety_buffer"]
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