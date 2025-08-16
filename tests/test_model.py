#!/usr/bin/env python3
"""
Simple test script to verify model loading and generation works
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    backend: str = "huggingface"
    model_name: str = "openai/gpt-oss-20b"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 50  # Reduced for quick test
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 7

def test_model_loading(cfg):
    """Test if we can load the model successfully"""
    print(f"🔄 Testing model loading...")
    print(f"Model: {cfg.model_name}")
    print(f"Device: {cfg.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        print("✅ Tokenizer loaded successfully")
        
        print("Loading model...")
        dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, 
            torch_dtype=dtype, 
            device_map="auto"
        )
        model.eval()
        print(f"✅ Model loaded successfully on device: {model.device}")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def test_generation(tokenizer, model, cfg):
    """Test a simple generation"""
    print(f"\n🔄 Testing generation...")
    
    # Simple test message
    test_messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    try:
        # Format for chat
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "User: What is 2+2?\nAssistant:"
        
        print(f"Prompt: {text}")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate
        torch.manual_seed(cfg.seed)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_only = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        
        print(f"✅ Generation successful!")
        print(f"Generated text: {generated_only}")
        print(f"Full response length: {len(full_text)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    cfg = ModelConfig()
    
    print("🧪 Starting model test...")
    print("=" * 50)
    
    # Test loading
    tokenizer, model = test_model_loading(cfg)
    
    if tokenizer is not None and model is not None:
        # Test generation
        success = test_generation(tokenizer, model, cfg)
        
        if success:
            print("\n🎉 All tests passed! Model is working correctly.")
        else:
            print("\n⚠️  Model loaded but generation failed.")
    else:
        print("\n❌ Model loading failed. Check your model path and dependencies.")
    
    print("=" * 50)