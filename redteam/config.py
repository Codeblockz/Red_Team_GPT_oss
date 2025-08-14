from dataclasses import dataclass
from typing import Literal

Backend = Literal["huggingface"]

@dataclass
class ModelConfig:
    backend: Backend = "huggingface"
    model_name: str = "gpt-oss-20b"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 7
    use_harmony_chat_template: bool = True
    add_special_tokens: bool = True

@dataclass
class RunConfig:
    out_dir: str = "artifacts"
    limit_attempts: int = 200
    exploration_policy: str = "ucb1"

@dataclass
class JudgeConfig:
    threshold: float = 0.6

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    run: RunConfig = RunConfig()
    judge: JudgeConfig = JudgeConfig()
