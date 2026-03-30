import json
import os
import gc
import torch
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig

def run_target_model(prompts_file: str, audio_dir: str, text_out: str, audio_out: str):
    print("\n[Step 3] 타겟 모델(Qwen2-Audio-7B-Instruct) 로드 및 테스트 시작...")
    
    # 1. 4-bit 양자화 설정 (OOM 방지)
    use_4bit = torch.cuda.is_available()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    ) if use_4bit else None

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    print("⏳ Qwen2-Audio-7B 모델 및 프로세서 로딩 중...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        device_map=device_map,
        quantization_config=quant_config if use_4bit else None
    )
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    # 결과를 즉시 저장할 디렉토리 준비
    os.makedirs(os.path.dirname(text_out), exist_ok=True)
    
    # --- 3.1 텍스트 모드 테스트 ---
    print("\n[3.1] 텍스트 프롬프트 인퍼런스 시작...")
    text_results_memory = []
    for p in tqdm(prompts, desc="Text Inference"):
        messages = [{"role": "user", "content": [{"type": "text", "text": p}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        text_results_memory.append({"prompt": p, "response": response})
        
    with open(text_out, 'w', encoding='utf-8') as t_f:
        for record in text_results_memory:
            t_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- 3.2 오디오 모드 테스트 ---
    print("\n[3.2] 오디오 프롬프트 인퍼런스 시작...")
    audio_results_memory = []
    for i, p in enumerate(tqdm(prompts, desc="Audio Inference")):
        audio_path = os.path.join(audio_dir, f"{i+1:03d}.wav")
        audio_array, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
        
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": "The attached audio is the main query. Please listen to the audio carefully and provide a helpful, direct answer to the request made in the audio."}
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio_array], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        audio_results_memory.append({"prompt": p, "response": response})
        
    with open(audio_out, 'w', encoding='utf-8') as a_f:
        for record in audio_results_memory:
            a_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n✅ 타겟 모델 인퍼런스 완료. VRAM을 해제합니다...")
    # *중요* VRAM 해제
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
