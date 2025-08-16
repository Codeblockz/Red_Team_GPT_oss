#!/usr/bin/env python3
"""
Ollama backend integration for the red-teaming notebook
This allows using Ollama's GPT-OSS model as a drop-in replacement for HuggingFace
"""

import subprocess
import json
import time
from dataclasses import asdict
from typing import List, Dict

class OllamaRunner:
    """Ollama model runner compatible with the red-teaming notebook"""
    
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
            # Check if Ollama is running
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"❌ Ollama command failed: {result.stderr}")
                return False
            
            # Check if our model is available
            if self.ollama_model.split(':')[0] not in result.stdout:
                print(f"❌ Model {self.ollama_model} not found in Ollama")
                print(f"Available models: {result.stdout}")
                print(f"💡 Try: ollama pull {self.ollama_model}")
                return False
            
            print(f"✅ Found {self.ollama_model} in Ollama")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Ollama command timed out")
            return False
        except FileNotFoundError:
            print("❌ Ollama not found in PATH")
            return False
        except Exception as e:
            print(f"❌ Ollama check failed: {e}")
            return False

    def _format_messages_for_ollama(self, messages: List[Dict]) -> str:
        """Format messages for Ollama chat interface"""
        # Build conversation string
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
        
        # Add final prompt for assistant response
        conversation_parts.append("Assistant:")
        
        return "\n\n".join(conversation_parts)

    def generate_chat(self, messages: List[Dict]) -> Dict:
        """Generate response using Ollama (compatible with HFRunner interface)"""
        start_time = time.time()
        
        # Format prompt
        prompt_text = self._format_messages_for_ollama(messages)
        
        try:
            # Prepare Ollama command
            cmd = [
                'ollama', 'run', self.ollama_model,
                '--', prompt_text
            ]
            
            # Add generation parameters if supported
            # Note: Ollama has different parameter names than transformers
            
            print(f"🤖 Generating with Ollama...")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama generation failed: {result.stderr}")
            
            # Extract the response
            response_text = result.stdout.strip()
            
            # Build full conversation text
            full_text = prompt_text + " " + response_text
            
            generation_time = time.time() - start_time
            print(f"✅ Generated {len(response_text)} chars in {generation_time:.1f}s")
            
            # Return in HFRunner-compatible format
            return {
                "prompt_text": prompt_text,
                "full_text": full_text,
                "gen_text": response_text,
                "input_ids": [[1, 2, 3]],  # Mock token IDs
                "generated_ids": [[1, 2, 3, 4, 5]],  # Mock generated IDs
                "model": asdict(self.cfg.model),
                "gen_params": {
                    "max_new_tokens": self.cfg.model.max_new_tokens,
                    "temperature": self.cfg.model.temperature,
                    "top_p": self.cfg.model.top_p,
                    "seed": self.cfg.model.seed,
                    "backend": "ollama",
                    "ollama_model": self.ollama_model
                },
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Ollama generation timed out after 120 seconds")
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")

def test_ollama_backend(cfg):
    """Test the Ollama backend"""
    print("🧪 Testing Ollama Backend")
    print("=" * 40)
    
    try:
        # Initialize backend
        runner = OllamaRunner(cfg)
        
        # Test generation
        test_messages = [
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ]
        
        result = runner.generate_chat(test_messages)
        
        print(f"✅ Test successful!")
        print(f"📝 Response: {result['gen_text']}")
        print(f"🎯 Backend ready for red-teaming!")
        
        return runner
        
    except Exception as e:
        print(f"❌ Ollama backend test failed: {e}")
        return None

if __name__ == "__main__":
    # Test with default config
    from dataclasses import dataclass
    from typing import Literal
    
    Backend = Literal["huggingface", "ollama"]
    
    @dataclass
    class ModelConfig:
        backend: Backend = "ollama"
        model_name: str = "openai/gpt-oss-20b"
        device: str = "cuda"
        dtype: str = "bfloat16"
        max_new_tokens: int = 256
        temperature: float = 0.2
        top_p: float = 0.9
        seed: int = 7
        use_harmony_chat_template: bool = True
        add_special_tokens: bool = True

    @dataclass
    class Config:
        model: ModelConfig = ModelConfig()
    
    cfg = Config()
    test_ollama_backend(cfg)