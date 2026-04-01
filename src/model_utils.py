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
            device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def get_hidden_states(self, messages, audio=None, target_layer: int = 15):
        pass

    @torch.no_grad()
    def get_hidden_states_and_response(self, messages, audio=None, target_layer: int = 15, max_new_tokens: int = 30):
        # Format input using processor
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if audio is not None:
            inputs = self.processor(text=text, audio=audio, return_tensors="pt", padding=True, sampling_rate=16000)
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
        
        inputs = inputs.to(self.model.device)
        
        if audio is None and "audio_features" in inputs:
            del inputs["audio_features"]
        
        # 1. Forward pass (for hidden states)
        forward_outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = forward_outputs.hidden_states[target_layer] # shape: (1, seq_len, hidden_size)
        
        input_ids_list = inputs["input_ids"][0].tolist()
        im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        if im_end_token_id is not None and im_end_token_id in input_ids_list:
            last_im_end_idx = len(input_ids_list) - 1 - input_ids_list[::-1].index(im_end_token_id)
            target_token_idx = last_im_end_idx - 1
        else:
            target_token_idx = -1
            
        target_hidden_state = hidden_states[:, target_token_idx, :].cpu().numpy()[0]
        
        # 2. Generation (for text output)
        generate_outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = generate_outputs[:, input_length:]
        response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return target_hidden_state, response_text
            target_token_idx = -1
            
        return hidden_states[:, target_token_idx, :].cpu().numpy()
