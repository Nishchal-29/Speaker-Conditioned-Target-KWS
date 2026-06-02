import os
import sys
import asyncio
import argparse
import random
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import edge_tts

# 20 diverse English voices spanning gender, accent, age
def get_active_english_voices():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Fetch the live list of voices from Microsoft
        all_voices = loop.run_until_complete(edge_tts.list_voices())
        
        # Filter for only English voices (en-US, en-GB, en-AU, etc.)
        en_voices = [v['ShortName'] for v in all_voices if v['Locale'].startswith('en')]
        
        # Sort them to ensure deterministic behavior, then shuffle to get a diverse mix
        en_voices.sort()
        random.seed(42) # Fixed seed so you get the same 20 voices if you restart the script
        random.shuffle(en_voices)
        
        return en_voices
    finally:
        loop.close()

def pitch_shift(audio, sr, n_semitones):
    return librosa.effects.pitch_shift(
        y=audio, sr=sr, n_steps=n_semitones
    )

def time_stretch(audio, rate):
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return stretched

def augment_audio(audio, sr, n_augments=3):
    """
    Generate n_augments pitch/speed variants of the base audio.

    Each variant gets:
        - Random pitch shift: U(-2, +2) semitones
        - Random time stretch: U(0.8, 1.2) rate

    Returns: list of (augmented_audio, aug_label) tuples
    """
    augmented = []
    for i in range(n_augments):
        # Random pitch shift in semitones
        pitch_semitones = random.uniform(-2.0, 2.0)
        # Random time stretch factor
        speed_factor = random.uniform(0.8, 1.2)

        aug = audio.copy()

        # Apply pitch shift
        aug = pitch_shift(aug, sr, pitch_semitones)

        # Apply time stretch
        aug = time_stretch(aug, speed_factor)

        label = f"p{pitch_semitones:+.1f}_s{speed_factor:.2f}"
        augmented.append((aug, label))

    return augmented

async def generate_tts(text, voice, output_path):
    """
    Generate TTS audio using edge-tts.

    Args:
        text: Word or phrase to synthesise.
        voice: Edge-TTS voice name.
        output_path: Path to save the WAV file.
    """
    communicate = edge_tts.Communicate(text, voice)
    # edge-tts outputs MP3 by default, we'll convert to WAV
    mp3_path = output_path.replace('.wav', '.mp3')
    await communicate.save(mp3_path)

    # Convert MP3 to 16kHz mono WAV
    if os.path.exists(mp3_path):
        audio, sr = librosa.load(mp3_path, sr=16000, mono=True)
        sf.write(output_path, audio, 16000)
        os.remove(mp3_path)
        return audio
    return None

def generate_tts_sync(text, voice, output_path):
    """Synchronous wrapper for async TTS generation."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(generate_tts(text, voice, output_path))
    finally:
        loop.close()

def generate_word_corpus(words, output_dir, n_voices=20, n_augments=3,
                         sample_rate=16000):
    """
    Generate the complete TTS corpus for the Phonetic Explosion strategy.

    For each word:
        - Generate base utterances with n_voices TTS voices
        - Apply n_augments pitch/speed perturbations per voice
        - Total: n_voices * (1 + n_augments) files per word
    """
    live_voices = get_active_english_voices()
    voices = live_voices[:n_voices]
    if len(voices) < n_voices:
        print(f"Warning: Only {len(voices)} English voices are currently online.")
    total_per_word = n_voices * (1 + n_augments)

    print(f"TTS CORPUS GENERATION")
    print(f"  Words: {len(words)}")
    print(f"  Voices: {n_voices}")
    print(f"  Augments per voice: {n_augments}")
    print(f"  Files per word: {total_per_word}")
    print(f"  Total files: {len(words) * total_per_word:,}")
    print(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for word_idx, word in enumerate(words):
        word_dir = os.path.join(output_dir, word.lower().strip())
        os.makedirs(word_dir, exist_ok=True)

        # Check if word already has enough files
        existing = len([f for f in os.listdir(word_dir) if f.endswith('.wav')])
        if existing >= total_per_word:
            if word_idx % 500 == 0:
                print(f"  [{word_idx + 1}/{len(words)}] {word}: "
                      f"already has {existing} files, skipping")
            continue

        if word_idx % 100 == 0:
            print(f"  [{word_idx + 1}/{len(words)}] Generating: {word}")

        for voice_idx, voice in enumerate(voices):
            voice_short = f"spk{voice_idx + 1:03d}"
            base_path = os.path.join(word_dir, f"{voice_short}_utt1.wav")

            # Skip if base file exists
            if os.path.exists(base_path):
                # Still try to generate augments
                try:
                    base_audio, _ = librosa.load(base_path, sr=sample_rate,
                                                  mono=True)
                except Exception:
                    continue
            else:
                # Generate base TTS
                try:
                    base_audio = generate_tts_sync(word, voice, base_path)
                    if base_audio is None:
                        continue
                except Exception as e:
                    if word_idx < 5:  # Only log first few errors
                        print(f"    TTS error for {word}/{voice}: {e}")
                    continue

            # Generate augmented variants
            augmented = augment_audio(base_audio, sample_rate, n_augments)

            for aug_idx, (aug_audio, aug_label) in enumerate(augmented):
                aug_path = os.path.join(
                    word_dir,
                    f"{voice_short}_aug{aug_idx + 1}_{aug_label}.wav"
                )
                if not os.path.exists(aug_path):
                    try:
                        # Ensure consistent length (pad/crop to 1.5s max)
                        max_len = int(1.5 * sample_rate)
                        if len(aug_audio) > max_len:
                            aug_audio = aug_audio[:max_len]

                        sf.write(aug_path, aug_audio, sample_rate)
                    except Exception:
                        pass

    # Final statistics
    total_files = 0
    total_words = 0
    for word_dir_name in os.listdir(output_dir):
        word_path = os.path.join(output_dir, word_dir_name)
        if os.path.isdir(word_path):
            n_files = len([f for f in os.listdir(word_path)
                          if f.endswith('.wav')])
            if n_files > 0:
                total_words += 1
                total_files += n_files

    print(f"GENERATION COMPLETE")
    print(f"  Words with audio: {total_words}")
    print(f"  Total audio files: {total_files:,}")
    print(f"  Average files per word: {total_files / max(total_words, 1):.1f}")

def load_wordlist(path):
    """Load a word list from a text file (one word per line)."""
    with open(path, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


# def generate_default_wordlist(output_path, n_words=10000):
#     """
#     Generate a default word list from NLTK corpus if available,
#     otherwise provide a minimal starter list.
#     """
#     try:
#         import nltk
#         nltk.download('words', quiet=True)
#         from nltk.corpus import words as nltk_words

#         all_words = nltk_words.words()
#         # Filter: 3–12 characters, alphabetic only
#         filtered = [w.lower() for w in all_words
#                     if 3 <= len(w) <= 12 and w.isalpha()]
#         # Deduplicate
#         filtered = list(set(filtered))
#         random.seed(42)
#         random.shuffle(filtered)
#         selected = sorted(filtered[:n_words])

#         with open(output_path, 'w') as f:
#             for w in selected:
#                 f.write(w + '\n')

#         print(f"Generated word list: {len(selected)} words → {output_path}")
#         return selected

#     except ImportError:
#         print("NLTK not available")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TTS corpus for Phonetic Explosion training"
    )
    parser.add_argument("--wordlist", type=str, default=None,
                       help="Path to word list file (one word per line)")
    parser.add_argument("--word", type=str, default=None,
                       help="Generate for a single word (for testing)")
    parser.add_argument("--output", type=str, default="../data/tts_words",
                       help="Output directory for TTS corpus")
    parser.add_argument("--voices", type=int, default=20,
                       help="Number of TTS voices to use (max 20)")
    parser.add_argument("--augments", type=int, default=3,
                       help="Number of augmented variants per voice")
    # parser.add_argument("--generate_wordlist", action="store_true",
    #                    help="Auto-generate a word list from NLTK")
    parser.add_argument("--n_words", type=int, default=10000,
                       help="Number of words for auto-generated list")

    args = parser.parse_args()

    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)

    if args.word:
        # Single word mode (for testing)
        generate_word_corpus(
            [args.word], args.output,
            n_voices=args.voices, n_augments=args.augments
        )

    elif args.wordlist:
        words = load_wordlist(args.wordlist)
        print(f"Loaded {len(words)} words from {args.wordlist}")
        generate_word_corpus(
            words, args.output,
            n_voices=args.voices, n_augments=args.augments
        )

    else:
        print("Specify --wordlist, --word, or --generate_wordlist")
        print("\nQuick start:")
        print("  python generate_tts_data.py --generate_wordlist --n_words 100")
        print("  python generate_tts_data.py --word activate")
        parser.print_help()