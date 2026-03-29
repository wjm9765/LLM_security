import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleHFModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading {model_name} on {self.device} in bfloat16/float16...")
        
        # OOM 방지를 위해 기본적으로 bfloat16 사용 (m1/m2 mac이나 구형 GPU 호환을 위해 float16을 쓸 수도 있음)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # device_map="auto"로 설정 시 accelerate가 가용 VRAM에 맞게 모델을 분산 로드해 줍니다.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto"
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # 모델에 맞는 채팅 템플릿(Chat template)을 적용하려면 여기서 전처리 과정을 커스텀할 수 있습니다.
        # 가장 단순한 형태로는 텍스트 그대로 인코딩합니다.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 프롬프트를 제외한 새로 생성된 텍스트 부분만 추출
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
