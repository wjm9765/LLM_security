import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np

class TargetModelWrapper:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        self.model.eval()

    @torch.no_grad()
    def get_hidden_states(self, messages, audio=None, target_layer: int = 15):
        # Format input using processor
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Audio handling (if present in messages, prepare it)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer] # shape: (batch_size, seq_len, hidden_size)
        
        # Extract target token (t_inst) - the token just before <|im_end|> of the user turn.
        # This is a simplified extraction; proper exact matching requires looking at input_ids
        # Assuming Qwen2-Audio ChatML format:
        # User message usually ends with <|im_end|>. The last token of the instruction is just before generated response prompt.
        # For a single sequence, generation prompt usually ends at the last token of input.
        # Thus, we take the last token of the input as a proxy for the end of the instruction prompt.
        last_token_idx = -1
        return hidden_states[:, last_token_idx, :].cpu().numpy()
