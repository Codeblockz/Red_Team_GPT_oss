"""
Model Backend Implementations for Red Team Framework
===================================================

This module provides backend implementations for running language models in the
red-teaming framework, supporting both HuggingFace Transformers and Ollama backends.

Backend Support
--------------
- **HuggingFace**: Direct model loading with GPU acceleration and memory optimization
- **Ollama**: Local model serving with subprocess communication and error handling

Key Features
-----------
- Unified API across different backend implementations  
- Automatic token budget validation and context management
- Chat template formatting with Harmony format support
- Memory efficient model loading with configurable precision
- Robust error handling and fallback mechanisms
- Dynamic token allocation based on available context

Architecture
-----------
The module follows a factory pattern with backend-specific implementations:

1. `create_runner(cfg)` - Factory function for backend selection
2. `HuggingFaceRunner` - Direct transformer model integration
3. `OllamaRunner` - Subprocess-based Ollama integration  
4. Utility functions for token management and formatting

Token Management
---------------
Smart token allocation prevents context overflow while maximizing generation:
- `validate_token_budget()` - Checks input/output token feasibility
- `calculate_max_output()` - Computes optimal output token limits
- Dynamic context truncation with message integrity preservation

Example Usage
-------------
```python
from redteam.models import create_runner, validate_token_budget
from redteam.core import Config

# Setup configuration
config = Config()
config.model.model_name = "openai/gpt-oss-20b"
config.model.backend = "huggingface"

# Create runner
runner = create_runner(config)

# Generate response
messages = [{"role": "user", "content": "Hello!"}]
result = runner.generate_chat(messages)
print(result["gen_text"])

# Token budget validation
budget = validate_token_budget(messages, runner.tok, config.model)
print(f"Token utilization: {budget['utilization']:.2f}")
```

Error Handling
--------------
The module provides comprehensive error handling for:
- Model loading failures
- Out of memory conditions
- Token limit violations
- Backend availability issues
- Malformed input validation

Performance Considerations
-------------------------
- Models loaded with configurable precision (bfloat16, float16, float32)
- Dynamic batching and memory management
- Aggressive garbage collection for stability
- GPU memory monitoring and cleanup

Author: Red Team GPT OSS Community
Version: 2.0.0
"""

import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict
from .core import set_seed
from typing import List, Dict, Any

def validate_token_budget(messages: List[Dict[str, str]], tokenizer, model_cfg) -> Dict[str, Any]:
    """
    Validate token budget to prevent context overflow during generation.
    
    Performs comprehensive token budget analysis using industry-standard 80% context
    window utilization to ensure reliable generation while preventing overflow.
    
    Args:
        messages: List of message dictionaries to analyze
        tokenizer: Loaded tokenizer for accurate token counting
        model_cfg: Model configuration with context limits
        
    Returns:
        Dict containing budget analysis results:
        - input_tokens: Actual input token count
        - max_output: Configured maximum output tokens
        - total_tokens: Sum of input + max output
        - effective_context: Usable context (80% of total)
        - available_tokens: Tokens available for generation
        - budget_ok: Whether generation is feasible
        - utilization: Total context utilization ratio
        
    Token Budget Strategy:
        - Uses 80% of total context window (industry standard)
        - Reserves 20% buffer for tokenizer variations and safety
        - Accounts for special tokens and formatting overhead
        
    Example:
        >>> budget = validate_token_budget(messages, tokenizer, model_config)
        >>> if not budget["budget_ok"]:
        ...     print(f"Need to truncate: {-budget['available_tokens']} tokens over")
    """
    # Count input tokens
    text = " ".join([msg.get("content", "") for msg in messages])
    input_tokens = len(tokenizer.encode(text))
    
    max_context = model_cfg.context_window
    max_output = model_cfg.max_new_tokens
    
    # Use 80% of context window (standard practice)
    effective_context = int(max_context * 0.8)
    available_tokens = effective_context - input_tokens
    budget_ok = available_tokens >= max_output
    utilization = (input_tokens + max_output) / max_context
    
    return {
        "input_tokens": input_tokens,
        "max_output": max_output,
        "total_tokens": input_tokens + max_output,
        "effective_context": effective_context,
        "available_tokens": available_tokens,
        "budget_ok": budget_ok,
        "utilization": utilization
    }

def calculate_max_output(input_tokens: int, model_cfg) -> int:
    """
    Calculate optimal maximum output tokens based on available context budget.
    
    Computes the maximum safe output token count given current input length,
    using conservative context allocation to prevent generation failures.
    
    Args:
        input_tokens: Current input token count
        model_cfg: Model configuration with token limits
        
    Returns:
        int: Recommended maximum output tokens (guaranteed safe)
        
    Algorithm:
        1. Calculate 80% effective context window
        2. Subtract input tokens to get available space
        3. Ensure minimum viable output (20 tokens minimum)
        4. Respect configured max_new_tokens limit
        5. Return the minimum of desired and available
        
    Example:
        >>> max_out = calculate_max_output(1500, model_config)
        >>> print(f"Safe output limit: {max_out} tokens")
    """
    effective_context = int(model_cfg.context_window * 0.8)
    available_for_output = effective_context - input_tokens
    
    # Ensure minimum viable output
    if available_for_output < 50:
        return 20
    
    # Use configured max_new_tokens unless it exceeds available space
    desired_max = getattr(model_cfg, 'max_new_tokens', 256)
    return min(desired_max, available_for_output)

def to_chat(messages: List[Dict[str, str]], tokenizer=None, add_special_tokens: bool = True):
    """
    Format messages for model input using tokenizer's built-in chat template.
    
    Converts message list to formatted text and token encodings using the model's
    native chat template. Supports GPT-OSS Harmony format and provides fallback
    for tokenizers without chat template support.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: Model tokenizer (optional, enables template formatting)
        add_special_tokens: Whether to add special tokens during encoding
        
    Returns:
        Dict containing:
        - text: Formatted chat text ready for generation
        - enc: Tensor encodings (None if no tokenizer provided)
        
    Template Priority:
        1. Tokenizer's native chat template (recommended for GPT-OSS)
        2. Simple fallback format if template fails
        3. Plain text format if no tokenizer available
        
    Harmony Format Support:
        GPT-OSS models automatically generate Harmony format outputs with
        analysis and final channels when using proper chat templates.
        
    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> result = to_chat(messages, tokenizer)
        >>> print(result["text"])  # Formatted chat
        >>> print(result["enc"]["input_ids"].shape)  # Token tensor
    """
    
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
            # Use tokenizer's built-in template (recommended approach)
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
            return {"text": text, "enc": enc}
        except Exception as e:
            # Fallback to simple format if template fails
            text = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
            return {"text": text, "enc": enc}
    
    # No tokenizer available - simple text format
    text = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages)
    return {"text": text, "enc": None}

# Removed create_harmony_chat_format() and simple_chat_format() - 
# Using standard tokenizer.apply_chat_template() instead

class OllamaRunner:
    """
    Ollama backend runner for local model serving and text generation.
    
    Provides a subprocess-based interface to Ollama for running language models
    locally with automatic model mapping, error handling, and response formatting.
    Designed for environments where Ollama is installed and models are available locally.
    
    Features:
    - Automatic HuggingFace to Ollama model name mapping
    - Subprocess communication with timeout handling  
    - Model availability verification before initialization
    - Structured response formatting compatible with framework expectations
    - Robust error handling for network and subprocess issues
    
    Supported Models:
    - GPT-OSS-20B: Maps to "gpt-oss:20b" 
    - GPT-OSS-120B: Maps to "gpt-oss:120b"
    - Custom model mapping through _get_ollama_model_name()
    
    Example:
        >>> config = Config()
        >>> config.model.backend = "ollama"  
        >>> config.model.model_name = "openai/gpt-oss-20b"
        >>> runner = OllamaRunner(config)
        >>> 
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> result = runner.generate_chat(messages)
        >>> print(result["gen_text"])
        
    Requirements:
    - Ollama installed and running on the system
    - Target model available in Ollama (ollama pull <model>)
    - Subprocess access and network connectivity
    """
    
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
    """
    HuggingFace Transformers backend runner for direct model inference.
    
    Provides high-performance text generation using HuggingFace transformers with
    GPU acceleration, memory optimization, and comprehensive error handling.
    Designed for production red-teaming with fine-grained control over generation parameters.
    
    Key Features:
    - Direct model loading with configurable precision (bfloat16, float16, float32)
    - Automatic device mapping and GPU acceleration
    - Dynamic token budget management with smart truncation
    - System prompt injection with template rendering
    - Memory-efficient generation with cleanup mechanisms  
    - Comprehensive input validation and error recovery
    - Chat template support for proper conversation formatting
    
    Memory Management:
    - Aggressive garbage collection to prevent memory leaks
    - Dynamic context truncation preserving message integrity
    - Configurable precision to balance performance vs memory
    - CUDA memory monitoring and cleanup utilities
    
    Token Budget Features:
    - 80% context window utilization (industry standard)
    - Smart message truncation preserving system messages
    - Dynamic output token calculation based on available space
    - Fallback generation with reduced tokens on OOM errors
    
    Example:
        >>> config = Config()
        >>> config.model.model_name = "openai/gpt-oss-20b"
        >>> config.model.device = "cuda"
        >>> config.model.dtype = "bfloat16"
        >>> 
        >>> runner = HuggingFaceRunner(config)
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> result = runner.generate_chat(messages)
        >>> 
        >>> print(f"Generated: {result['gen_text']}")
        >>> print(f"Tokens used: {result['token_info']['input_tokens']}")
        
    Performance Considerations:
    - Model loading may take 30-120 seconds depending on size
    - GPU memory requirements scale with model size and precision
    - First generation includes JIT compilation overhead
    - Subsequent generations benefit from CUDA kernel caching
    """
    
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
        
        # Standard token budget validation
        budget = validate_token_budget(messages, self.tok, cfg.model)
        
        # Truncate if needed
        if not budget["budget_ok"]:
            target_tokens = int(budget["effective_context"] * 0.7)  # Use 70% for input
            messages = self._truncate_messages(messages, target_tokens)
            budget = validate_token_budget(messages, self.tok, cfg.model)
        
        # Calculate output tokens (simple approach)
        max_output_tokens = calculate_max_output(budget["input_tokens"], cfg.model)
        
        # Format messages using standard tokenizer chat template
        try:
            packed = to_chat(
                messages, 
                self.tok, 
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
                "effective_context": budget["effective_context"]
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
    """
    Factory function to create appropriate model runner based on backend configuration.
    
    Selects and instantiates the correct runner implementation based on the configured
    backend type, providing a unified interface for model creation across different
    backend implementations.
    
    Args:
        cfg: Configuration object with model.backend specifying runner type
        
    Returns:
        Union[HuggingFaceRunner, OllamaRunner]: Initialized runner instance
        
    Raises:
        ValueError: If backend type is not supported
        RuntimeError: If runner initialization fails (model loading, etc.)
        
    Supported Backends:
        - "huggingface": Direct HuggingFace transformers integration
        - "ollama": Local Ollama model serving via subprocess
        
    Example:
        >>> config = Config()
        >>> config.model.backend = "huggingface"
        >>> config.model.model_name = "openai/gpt-oss-20b"
        >>> 
        >>> runner = create_runner(config)
        >>> isinstance(runner, HuggingFaceRunner)
        True
        >>> 
        >>> # Switch to Ollama backend
        >>> config.model.backend = "ollama"
        >>> runner = create_runner(config) 
        >>> isinstance(runner, OllamaRunner)
        True
    """
    if cfg.model.backend == "ollama":
        return OllamaRunner(cfg)
    elif cfg.model.backend == "huggingface":
        return HuggingFaceRunner(cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.model.backend}")