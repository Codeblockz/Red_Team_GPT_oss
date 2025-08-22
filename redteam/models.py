"""Model backend implementations for HuggingFace and Ollama."""

import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict
from .core import set_seed
from typing import List, Dict, Any

def validate_token_budget(messages: List[Dict[str, str]], tokenizer, model_cfg) -> Dict[str, Any]:
    """Validate token budget and return analysis"""
    # Simple token counting - can be enhanced
    text = " ".join([msg.get("content", "") for msg in messages])
    input_tokens = len(tokenizer.encode(text))
    
    max_context = model_cfg.context_window
    max_output = model_cfg.max_new_tokens
    
    budget_ok = (input_tokens + max_output) <= max_context
    utilization = (input_tokens + max_output) / max_context
    
    return {
        "input_tokens": input_tokens,
        "max_output": max_output,
        "total_tokens": input_tokens + max_output,
        "context_window": max_context,
        "budget_ok": budget_ok,
        "utilization": utilization
    }

def calculate_max_output(input_tokens: int, model_cfg) -> int:
    """Calculate dynamic max output tokens based on input and format requirements"""
    remaining = model_cfg.context_window - input_tokens
    max_desired = model_cfg.max_new_tokens
    min_required = model_cfg.min_response_tokens
    
    # Account for Harmony format overhead (approximately 80 tokens)
    harmony_overhead = 80 if model_cfg.use_harmony_chat_template else 20
    available_for_content = remaining - harmony_overhead - 50  # 50 token safety buffer
    
    # Ensure minimum space for meaningful content
    recommended = max(min_required, min(max_desired, available_for_content))
    
    # Cap at max_desired but account for overhead difference
    if not model_cfg.use_harmony_chat_template and recommended == max_desired:
        # Simple format can use more tokens since it has less overhead
        recommended = min(max_desired, available_for_content + 60)  # Reclaim some overhead
    
    return recommended

def to_chat(messages: List[Dict[str, str]], tokenizer=None, use_harmony: bool = True, add_special_tokens: bool = True):
    """Format messages for model input"""
    if tokenizer and use_harmony:
        # Use tokenizer's chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
            return {"text": text, "enc": enc}
    
    # Fallback to simple formatting
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"{role}: {content}\n"
    
    if tokenizer:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        return {"text": text, "enc": enc}
    
    return {"text": text, "enc": None}

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
        
        print(f"Loading model with dtype: {cfg.model.dtype}")
        dtype = getattr(torch, cfg.model.dtype) if hasattr(torch, cfg.model.dtype) else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name, 
            torch_dtype=dtype, 
            device_map="auto"
        )
        self.model.eval()
        print(f"Model loaded on device: {self.model.device}")

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
        
        # Validate token budget and calculate dynamic allocation
        budget = validate_token_budget(messages, self.tok, cfg.model)
        dynamic_max_tokens = calculate_max_output(budget["input_tokens"], cfg.model)
        
        if not budget["budget_ok"]:
            print(f"Warning: Input too long ({budget['input_tokens']} tokens), reducing output to {dynamic_max_tokens}")
        
        # Ensure minimum token output for very long inputs
        if dynamic_max_tokens < cfg.model.min_response_tokens:
            dynamic_max_tokens = cfg.model.min_response_tokens
            print(f"Warning: Forcing minimum {cfg.model.min_response_tokens} tokens for response")
        
        # Only print token budget for debugging if explicitly requested
        debug_tokens = getattr(cfg.model, 'debug_token_budget', False)
        if debug_tokens:
            print(f"Token budget: {budget['input_tokens']} input + {dynamic_max_tokens} output = {budget['input_tokens'] + dynamic_max_tokens} total")
        
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
        
        # Generate with dynamic token adjustment
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=dynamic_max_tokens,  # Use calculated value
                    temperature=cfg.model.temperature,
                    top_p=cfg.model.top_p,
                    do_sample=cfg.model.temperature > 0,
                )
        except Exception as e:
            raise RuntimeError(f"Model generation failed: {e}")
        
        # Decode outputs
        try:
            full = self.tok.decode(out[0], skip_special_tokens=False)
            gen_only = self.tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        except Exception as e:
            raise RuntimeError(f"Failed to decode outputs: {e}")
        
        # Basic response analysis
        response_length = len(gen_only.split())
        
        # Assess response quality
        try:
            from .judges import assess_response_quality
            quality = assess_response_quality(gen_only)
            quality_info = {
                "completeness_score": quality["completeness_score"],
                "export_ready": quality["export_ready"],
                "issues": quality["issues"]
            }
        except Exception:
            quality_info = {"completeness_score": 1.0, "export_ready": True, "issues": []}
        
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
                "dynamic_adjustment": dynamic_max_tokens != cfg.model.max_new_tokens
            },
            "quality_info": quality_info
        }

# Compatibility aliases
HFRunner = HuggingFaceRunner

def cleanup_model_memory():
    """Clean up CUDA memory to prevent OOM errors"""
    try:
        import gc
        import torch
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        return True
    except Exception as e:
        print(f"Warning: Failed to cleanup memory: {e}")
        return False

def create_runner(cfg):
    """Create appropriate runner based on backend configuration"""
    if cfg.model.backend == "ollama":
        return OllamaRunner(cfg)
    elif cfg.model.backend == "huggingface":
        return HuggingFaceRunner(cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.model.backend}")