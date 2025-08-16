#!/usr/bin/env python3
"""
Test script to verify GPT-OSS-20B model loading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import GPUtil
from dataclasses import dataclass
from typing import Literal

Backend = Literal["huggingface"]

@dataclass
class ModelConfig:
    backend: Backend = "huggingface"
    model_name: str = "openai/gpt-oss-20b"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 7
    use_harmony_chat_template: bool = True
    add_special_tokens: bool = True

def check_system_resources():
    """Check available system resources"""
    print("🔍 System Resource Check:")
    
    # CPU and RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"   RAM: {ram_available_gb:.1f}GB available / {ram_gb:.1f}GB total")
    
    # GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   CUDA devices: {gpu_count}")
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}, {gpu.memoryFree}MB free / {gpu.memoryTotal}MB total")
        except:
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {props.name}, ~{mem_total:.1f}GB total memory")
    else:
        print("   ⚠️  CUDA not available - will use CPU (very slow for 20B model)")
    
    return ram_available_gb >= 16, torch.cuda.is_available()

def test_tokenizer_loading(model_name: str):
    """Test tokenizer loading"""
    print(f"🔤 Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("   ✅ Tokenizer loaded successfully")
        
        # Test basic tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   ✅ Test tokenization: '{test_text}' -> {tokens['input_ids'].shape[1]} tokens")
        
        # Check if it has chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            print("   ✅ Chat template available")
        else:
            print("   ⚠️  No chat template - will use fallback format")
            
        return tokenizer
        
    except Exception as e:
        print(f"   ❌ Tokenizer loading failed: {e}")
        return None

def test_model_loading(cfg: ModelConfig, tokenizer):
    """Test model loading with different strategies"""
    print(f"🤖 Loading model: {cfg.model_name}")
    
    # Get dtype
    dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.bfloat16
    print(f"   Using dtype: {dtype}")
    
    try:
        # Strategy 1: Try with device_map="auto" (recommended for large models)
        print("   Attempting load with device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True  # May be needed for some models
        )
        model.eval()
        print("   ✅ Model loaded successfully with device_map='auto'")
        print(f"   Model device: {next(model.parameters()).device}")
        
        return model
        
    except Exception as e1:
        print(f"   ⚠️  device_map='auto' failed: {e1}")
        
        try:
            # Strategy 2: Try loading to CPU first, then move to GPU
            print("   Attempting load to CPU first...")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            
            # Try to move to GPU if available
            if torch.cuda.is_available() and cfg.device == "cuda":
                print("   Moving model to GPU...")
                model = model.cuda()
            
            model.eval()
            print(f"   ✅ Model loaded successfully and moved to {next(model.parameters()).device}")
            return model
            
        except Exception as e2:
            print(f"   ❌ Model loading failed: {e2}")
            print("   💡 Suggestions:")
            print("      - Ensure you have enough RAM/VRAM (16GB+ recommended)")
            print("      - Try running: huggingface-cli login (if private model)")
            print("      - Try: pip install --upgrade transformers accelerate")
            return None

def test_generation(model, tokenizer, cfg: ModelConfig):
    """Test basic generation"""
    print("🎭 Testing generation...")
    
    # Simple test messages
    test_messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    try:
        # Format messages
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback format
            text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"   Prompt: {text[:100]}...")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available() and cfg.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Short test
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        print(f"   ✅ Generation successful!")
        print(f"   Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 GPT-OSS-20B Model Loading Test")
    print("=" * 50)
    
    # Check system resources
    ram_ok, cuda_ok = check_system_resources()
    
    if not ram_ok:
        print("⚠️  Warning: Low RAM detected. Model loading may fail.")
    
    print()
    
    # Create config
    cfg = ModelConfig()
    if not cuda_ok:
        cfg.device = "cpu"
        print("🔄 Switching to CPU mode")
    
    print()
    
    # Test tokenizer
    tokenizer = test_tokenizer_loading(cfg.model_name)
    if not tokenizer:
        print("❌ Cannot proceed without tokenizer")
        return False
    
    print()
    
    # Test model loading
    model = test_model_loading(cfg, tokenizer)
    if not model:
        print("❌ Cannot proceed without model")
        return False
    
    print()
    
    # Test generation
    gen_ok = test_generation(model, tokenizer, cfg)
    
    print()
    print("=" * 50)
    if gen_ok:
        print("🎉 All tests passed! GPT-OSS-20B is ready for red-teaming.")
        print("📝 You can now run the notebook with confidence.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return gen_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)