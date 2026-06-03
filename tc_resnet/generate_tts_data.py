import os
import asyncio
import random
import numpy as np
import librosa
import soundfile as sf
import edge_tts

TARGET_SAMPLES = int(1.5 * 16000)

async def get_active_english_voices():
    all_voices = await edge_tts.list_voices()
    en_voices = [v['ShortName'] for v in all_voices if v['Locale'].startswith('en')]
    en_voices.sort()
    random.seed(42) 
    random.shuffle(en_voices)
    return en_voices

def enforce_length(audio, target_length):
    """Center-pads with silence or truncates to ensure exact length."""
    current_length = len(audio)
    if current_length < target_length:
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')
    elif current_length > target_length:
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]
    return audio

def pitch_shift(audio, sr, n_semitones):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_semitones)

def time_stretch(audio, rate):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def augment_audio(audio, sr, n_augments=3):
    augmented = []
    for i in range(n_augments):
        pitch_semitones = random.uniform(-2.0, 2.0)
        speed_factor = random.uniform(0.8, 1.2)
        aug = audio.copy()
        aug = pitch_shift(aug, sr, pitch_semitones)
        aug = time_stretch(aug, speed_factor)
        aug = enforce_length(aug, TARGET_SAMPLES)
        label = f"p{pitch_semitones:+.1f}_s{speed_factor:.2f}"
        augmented.append((aug, label))
    return augmented

async def generate_tts(text, voice, output_path, retries=3):
    mp3_path = output_path.replace('.wav', '.mp3')
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice, rate="+0%")
            await communicate.save(mp3_path)
            
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                audio, sr = await asyncio.to_thread(librosa.load, mp3_path, sr=16000, mono=True)
                audio = enforce_length(audio, TARGET_SAMPLES)
                await asyncio.to_thread(sf.write, output_path, audio, 16000)
                os.remove(mp3_path)
                return audio
        except Exception:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            await asyncio.sleep(2 ** attempt)
    return None

async def process_word_voice(word, voice, voice_short, word_dir, sample_rate, n_augments, semaphore):
    async with semaphore:
        base_path = os.path.join(word_dir, f"{voice_short}_utt1.wav")
        if not os.path.exists(base_path):
            base_audio = await generate_tts(word, voice, base_path)
            if base_audio is None:
                return
        else:
            base_audio, _ = await asyncio.to_thread(librosa.load, base_path, sr=sample_rate, mono=True)

        augmented = await asyncio.to_thread(augment_audio, base_audio, sample_rate, n_augments)
        for aug_idx, (aug_audio, aug_label) in enumerate(augmented):
            aug_path = os.path.join(word_dir, f"{voice_short}_aug{aug_idx + 1}_{aug_label}.wav")
            if not os.path.exists(aug_path):
                await asyncio.to_thread(sf.write, aug_path, aug_audio, sample_rate)

async def generate_split(words, split_name, output_base, voices, n_augments):
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(15)
    tasks = []

    print(f"\nGenerating {split_name.upper()} Split ({len(words)} words)")
    for word in words:
        word_dir = os.path.join(split_dir, word.lower().strip())
        os.makedirs(word_dir, exist_ok=True)
        
        for voice_idx, voice in enumerate(voices):
            voice_short = f"spk{voice_idx + 1:03d}"
            tasks.append(process_word_voice(word, voice, voice_short, word_dir, 16000, n_augments, semaphore))

    if tasks:
        chunk_size = 500
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            await asyncio.gather(*chunk)
            print(f"[{split_name}] Completed {min(i+chunk_size, len(tasks))}/{len(tasks)} tasks")

async def main(wordlist_path, output_dir, n_augments=3):
    with open(wordlist_path, 'r') as f:
        words = list(set([line.strip().lower() for line in f if line.strip()]))
    
    random.seed(42)
    random.shuffle(words)

    # 80/10/10 Split
    n_train = int(len(words) * 0.8)
    n_val = int(len(words) * 0.1)
    
    train_words = words[:n_train]
    val_words = words[n_train:n_train+n_val]
    test_words = words[n_train+n_val:]
    live_voices = await get_active_english_voices()
    train_voices = live_voices[:-10] if len(live_voices) > 20 else live_voices
    val_test_voices = live_voices[-10:] if len(live_voices) > 20 else live_voices[:10]

    await generate_split(train_words, "train", output_dir, train_voices, n_augments)
    await generate_split(val_words, "val", output_dir, val_test_voices, n_augments=1) 
    await generate_split(test_words, "test", output_dir, val_test_voices, n_augments=1)

if __name__ == "__main__":
    asyncio.run(main("../data/master_words.txt", "../data/tts_corpus", n_augments=3))