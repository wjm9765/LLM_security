import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import numpy as np

class TargetModelWrapper:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
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
        # Pass sampling_rate to avoid WhisperFeatureExtractor warnings
        if audio is not None:
            inputs = self.processor(text=text, audio=audio, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer] # shape: (batch_size, seq_len, hidden_size)
        
        # Extract target token (t_inst) - the exact token just before <|im_end|> of the user turn (🎯).
        # With add_generation_prompt=True, the sequence ends with: ... 🎯<|im_end|>\n<|im_start|>assistant\n
        # So the last `<|im_end|>` in input_ids is exactly the one closing the user message.
        input_ids_list = inputs["input_ids"][0].tolist()
        im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        if im_end_token_id is not None and im_end_token_id in input_ids_list:
            # Find the last occurrence of <|im_end|> (which closes the user turn)
            last_im_end_idx = len(input_ids_list) - 1 - input_ids_list[::-1].index(im_end_token_id)
            target_token_idx = last_im_end_idx - 1
        else:
            # Fallback just in case
            target_token_idx = -1
            
        return hidden_states[:, target_token_idx, :].cpu().numpy()
