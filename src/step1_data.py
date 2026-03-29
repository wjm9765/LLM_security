import json
import os
import random
from datasets import load_dataset

def download_and_sample(output_file: str, num_samples: int = 100):
    print("\n[Step 1] AdvBench(Jailbreak) 데이터셋 다운로드 및 샘플링 시작...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # 유명한 AdvBench 데이터셋 로드 시도
        dataset = load_dataset("rubenland/advbench", split="train")
        prompts = dataset['prompt']
    except Exception as e:
        print(f"AdvBench 로드 실패 ({e}). 대안으로 PKU-SafeRLHF의 Harmful 데이터를 가져옵니다.")
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
        filtered_dataset = dataset.filter(lambda x: not x["is_response_0_safe"] and not x["is_response_1_safe"])
        # 프롬프트 추출
        prompts = filtered_dataset['prompt']

    # 중복 제거 및 무작위 샘플링
    unique_prompts = list(set(prompts))
    random.shuffle(unique_prompts)
    sampled = unique_prompts[:num_samples]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(sampled)}개의 악성 프롬프트를 성공적으로 저장했습니다: {output_file}")
    return sampled
