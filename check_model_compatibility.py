#!/usr/bin/env python3
"""
Quick compatibility check for GPT-OSS-20B on this system
"""

import torch
import psutil
from transformers import AutoTokenizer
import subprocess
import sys

def check_system_specs():
    """Check if system can handle GPT-OSS-20B"""
    print("🔍 System Compatibility Check for GPT-OSS-20B")
    print("=" * 50)
    
    # Memory check
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_available = psutil.virtual_memory().available / (1024**3)
    print(f"📊 RAM: {ram_available:.1f}GB available / {ram_gb:.1f}GB total")
    
    # GPU check
    cuda_available = torch.cuda.is_available()
    print(f"🖥️  CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"🎮 GPU count: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name}")
            print(f"   Memory: {mem_gb:.1f}GB")
            
            # Check current memory usage
            try:
                torch.cuda.empty_cache()
                mem_free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                mem_free_gb = mem_free / (1024**3)
                print(f"   Available: ~{mem_free_gb:.1f}GB")
            except:
                print(f"   Available: ~{mem_gb:.1f}GB (estimated)")
    
    return {
        'ram_gb': ram_gb,
        'ram_available': ram_available,
        'cuda_available': cuda_available,
        'gpu_memory': props.total_memory / (1024**3) if cuda_available and gpu_count > 0 else 0
    }

def check_model_requirements():
    """Check model-specific requirements"""
    print(f"\n📋 GPT-OSS-20B Requirements:")
    print(f"   Model size: ~21B parameters")
    print(f"   Memory needed (bf16): ~42GB")
    print(f"   Memory needed (MXFP4): ~16GB (with quantization)")
    print(f"   Recommended: 32GB+ GPU memory or 64GB+ system RAM")

def test_tokenizer_only():
    """Test if we can at least load the tokenizer"""
    print(f"\n🔤 Testing tokenizer access...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        print("   ✅ Tokenizer loads successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        
        # Test basic functionality
        test_text = "Hello world"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   ✅ Tokenization works: '{test_text}' -> {tokens['input_ids'].shape[1]} tokens")
        
        if hasattr(tokenizer, 'apply_chat_template'):
            print("   ✅ Chat template available")
        else:
            print("   ⚠️  No chat template (will use fallback)")
            
        return True
    except Exception as e:
        print(f"   ❌ Tokenizer failed: {e}")
        return False

def check_ollama_integration():
    """Check if Ollama model can be used as alternative"""
    print(f"\n🦙 Checking Ollama integration...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            models = result.stdout
            print("   ✅ Ollama is available")
            
            if 'gpt-oss' in models:
                print("   ✅ GPT-OSS model found in Ollama")
                print("   💡 You can use Ollama API as an alternative backend")
            else:
                print("   ⚠️  GPT-OSS model not found in Ollama")
                print("   💡 Run: ollama pull gpt-oss:20b")
            
            return True
        else:
            print("   ⚠️  Ollama command failed")
            return False
    except FileNotFoundError:
        print("   ❌ Ollama not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  Ollama command timed out")
        return False
    except Exception as e:
        print(f"   ❌ Ollama check failed: {e}")
        return False

def provide_recommendations(specs):
    """Provide recommendations based on system specs"""
    print(f"\n💡 Recommendations:")
    
    gpu_mem = specs['gpu_memory']
    ram_available = specs['ram_available']
    
    # Can we load the full model?
    if gpu_mem >= 32:
        print("   ✅ GPU memory is sufficient for full model loading")
        print("   🚀 Use: transformers with device_map='auto'")
        return "full_gpu"
    elif gpu_mem >= 16:
        print("   ⚠️  GPU memory may be sufficient with quantization")
        print("   🎯 Try: transformers with quantization enabled")
        return "quantized_gpu"
    elif ram_available >= 50:
        print("   ⚠️  Could try CPU-only mode (will be slow)")
        print("   🐌 Warning: Very slow inference on CPU")
        return "cpu_only"
    else:
        print("   ❌ Insufficient memory for full model")
        print("   🦙 Recommendation: Use Ollama with quantized model")
        print("   🔄 Alternative: Use smaller model for testing")
        return "ollama_recommended"

def main():
    """Main compatibility check"""
    # System specs
    specs = check_system_specs()
    
    # Model requirements
    check_model_requirements()
    
    # Test tokenizer
    tokenizer_ok = test_tokenizer_only()
    
    # Check Ollama
    ollama_ok = check_ollama_integration()
    
    # Recommendations
    recommendation = provide_recommendations(specs)
    
    print(f"\n" + "=" * 50)
    print(f"🎯 FINAL ASSESSMENT:")
    
    if recommendation == "full_gpu":
        print("   ✅ Your system can handle GPT-OSS-20B with transformers")
        print("   📝 Proceed with the red-teaming notebook as configured")
    elif recommendation == "quantized_gpu":
        print("   ⚠️  Your system might handle GPT-OSS-20B with quantization")
        print("   🧪 Try the notebook, but be prepared for potential memory issues")
    elif recommendation == "cpu_only":
        print("   ⚠️  CPU-only mode possible but very slow")
        print("   🚀 Consider using Ollama for better performance")
    else:
        print("   ❌ Full model may not fit in memory")
        print("   🦙 Use Ollama or consider a smaller model for testing")
    
    print(f"\n📋 Next Steps:")
    if tokenizer_ok:
        print("   1. ✅ Tokenizer works - basic functionality confirmed")
    
    if recommendation in ["full_gpu", "quantized_gpu"]:
        print("   2. 🚀 Try running the red-teaming notebook")
        print("   3. 📊 Monitor memory usage during model loading")
    else:
        print("   2. 🦙 Consider setting up Ollama backend integration")
        print("   3. 🔄 Or use a smaller model for framework testing")
    
    return recommendation in ["full_gpu", "quantized_gpu"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)