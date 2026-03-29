import os
import asyncio
import edge_tts
from tqdm import tqdm

async def _verify_and_generate_audio(text, output_path, voice="en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_audios(prompts: list, output_dir: str):
    print("\n[Step 2] Edge-TTS를 활용한 텍스트 프롬프트 -> 오디오(WAV) 변환 시작...")
    os.makedirs(output_dir, exist_ok=True)
    
    async def run_all():
        tasks = []
        for i, prompt in enumerate(prompts):
            filename = os.path.join(output_dir, f"{i+1:03d}.wav")
            tasks.append(_verify_and_generate_audio(prompt, filename))
            
        # 순차적으로 처리하며 진행률 표시 (Too many requests 방지)
        for i, task in enumerate(tqdm(tasks, desc="TTS 생성 중")):
            await task

    asyncio.run(run_all())
    print(f"✅ 총 {len(prompts)}개의 오디오 프롬프트가 저장되었습니다: {output_dir}/")
