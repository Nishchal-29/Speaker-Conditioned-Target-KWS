# import os

# def get_validation_words():
#     """
#     Returns a curated list of exactly 200 words designed to stress-test
#     phoneme-level discrimination in the TC-ResNet embedding space.

#     Categories:
#       1. Minimal Pairs (40 words / 20 pairs)
#          Words differing by exactly one phoneme — if the encoder confuses
#          these, it lacks sub-phoneme resolution.

#       2. Rhyming Clusters (40 words / ~13 groups)
#          Words sharing tail phonemes — tests whether the encoder
#          distinguishes onset phonemes vs. relying on rhyme patterns.

#       3. Compound Confusions (30 words / 15 pairs)
#          Base words + compound forms — tests whether the encoder
#          treats "snow" and "snowball" as distinct.

#       4. Phonetically Dense (40 words)
#          Unusual consonant clusters, multi-syllabic words with
#          similar cadence patterns — maximal embedding space stress.

#       5. Cross-Language / Borrowed Words (20 words)
#          Words from other languages commonly used in English —
#          tests robustness to unfamiliar phoneme combinations.

#       6. Anchor Words (30 words)
#          Common wake-word candidates and simple phonetic anchors
#          that define the baseline geometry of the embedding space.
#     """

#     # --- Category 1: Minimal Pairs (40 words = 20 pairs) ---
#     minimal_pairs = [
#         # Vowel minimal pairs
#         "accept", "except",
#         "affect", "effect",
#         "adapt", "adopt",
#         "angel", "angle",
#         "dairy", "diary",
#         "desert", "dessert",
#         "eminent", "imminent",
#         "elicit", "illicit",
#         "moral", "morale",
#         "personal", "personnel",
#         # Consonant minimal pairs
#         "bat", "pat",
#         "bin", "pin",
#         "den", "ten",
#         "fan", "van",
#         "goat", "coat",
#         "light", "right",
#         "mail", "nail",
#         "seal", "zeal",
#         "sip", "ship",
#         "thin", "fin",
#     ]

#     # --- Category 2: Rhyming Clusters (40 words) ---
#     rhyming = [
#         # -ate cluster
#         "activate", "captivate", "motivate", "cultivate",
#         # -tion cluster
#         "station", "nation", "ration", "caution",
#         # -ado cluster
#         "tornado", "avocado", "bravado", "desperado",
#         # -ular cluster
#         "nebula", "fibula", "formula", "peninsula",
#         # -ight cluster
#         "midnight", "twilight", "spotlight", "moonlight",
#         # -ound cluster
#         "compound", "surround", "background", "playground",
#         # -ment cluster
#         "moment", "comment", "segment", "fragment",
#         # -ance cluster
#         "balance", "distance", "substance", "instance",
#         # -ible cluster
#         "possible", "terrible", "invisible", "incredible",
#         # -ness cluster
#         "darkness", "kindness", "awareness", "happiness",
#     ]

#     # --- Category 3: Compound Confusions (30 words = 15 pairs) ---
#     compounds = [
#         "snow", "snowball",
#         "base", "baseball",
#         "sun", "sunflower",
#         "fire", "fireplace",
#         "door", "doorbell",
#         "book", "bookmark",
#         "rain", "rainbow",
#         "star", "starlight",
#         "black", "blackberry",
#         "water", "waterfall",
#         "thunder", "thunderstorm",
#         "butter", "butterfly",
#         "green", "greenhouse",
#         "day", "daybreak",
#         "cup", "cupcake",
#     ]

#     # --- Category 4: Phonetically Dense (40 words) ---
#     phonetic_dense = [
#         "algorithm", "catastrophe", "electromagnetic", "hypothesis",
#         "kaleidoscope", "labyrinth", "metamorphosis", "onomatopoeia",
#         "photosynthesis", "quintessential", "revolutionary", "serendipity",
#         "thermometer", "unprecedented", "vulnerability", "xenomorph",
#         "abracadabra", "bibliography", "circumference", "demographics",
#         "encyclopedia", "fluorescent", "hieroglyphic", "infrastructure",
#         "juxtaposition", "kindergarten", "melancholy", "nomenclature",
#         "palindrome", "quadrilateral", "reconnaissance", "sophisticated",
#         "teleportation", "ubiquitous", "ventriloquist", "wanderlust",
#         "xylophone", "yesterday", "zeppelin", "archipelago",
#     ]

#     # --- Category 5: Cross-Language / Borrowed Words (20 words) ---
#     borrowed = [
#         "tsunami", "karate", "origami", "samurai",
#         "croissant", "entrepreneur", "rendezvous", "chauffeur",
#         "kindergarten", "wanderlust", "poltergeist", "zeitgeist",
#         "fiesta", "guerrilla", "mosquito", "tornado",
#         "avatar", "jungle", "nirvana", "yoga",
#     ]

#     # --- Category 6: Anchor Words (30 words) ---
#     anchors = [
#         "alexa", "computer", "jarvis", "cortana",
#         "activate", "navigate", "terminate", "initialize",
#         "listen", "record", "cancel", "confirm",
#         "volume", "silence", "pause", "resume",
#         "forward", "backward", "repeat", "delete",
#         "morning", "evening", "hello", "goodbye",
#         "open", "close", "start", "stop",
#         "weather", "music",
#     ]

#     # Combine all categories
#     all_words = (minimal_pairs + rhyming + compounds +
#                  phonetic_dense + borrowed + anchors)

#     # Deduplicate while preserving order
#     seen = set()
#     unique_words = []
#     for word in all_words:
#         w = word.lower().strip()
#         if w not in seen:
#             seen.add(w)
#             unique_words.append(w)

#     return unique_words


# def verify_list(words):
#     """Verify the validation word list meets all requirements."""
#     # Check for duplicates
#     if len(words) != len(set(words)):
#         dupes = [w for w in words if words.count(w) > 1]
#         raise ValueError(f"Duplicate words found: {set(dupes)}")

#     # Check exact count
#     if len(words) != 200:
#         raise ValueError(f"Expected exactly 200 words, got {len(words)}. "
#                          f"Adjust the word lists to hit exactly 200.")

#     # Check all lowercase
#     for w in words:
#         if w != w.lower():
#             raise ValueError(f"Word not lowercase: {w}")

#     # Check no empty strings
#     for w in words:
#         if not w.strip():
#             raise ValueError("Empty word found in list")

#     print(f"✓ Validation list verified: {len(words)} unique words, all lowercase")


# def main():
#     words = get_validation_words()

#     # Report statistics
#     print(f"Generated {len(words)} unique validation words")

#     if len(words) > 200:
#         print(f"  Trimming from {len(words)} to 200 (removing last {len(words) - 200})")
#         words = words[:200]
#     elif len(words) < 200:
#         deficit = 200 - len(words)
#         print(f"  WARNING: {deficit} words short of 200. Adding fillers...")
#         # Emergency fillers — phonetically diverse single words
#         fillers = [
#             "abyss", "breeze", "cactus", "dagger", "eclipse",
#             "falcon", "glacier", "horizon", "igloo", "javelin",
#             "kettle", "lantern", "magnet", "nucleus", "octopus",
#             "phantom", "quartz", "riddle", "saffron", "trumpet",
#             "umbrella", "vortex", "whistle", "zenith", "puzzle",
#             "crimson", "emerald", "sapphire", "cobalt", "bronze",
#         ]
#         for filler in fillers:
#             if filler.lower() not in set(words):
#                 words.append(filler.lower())
#                 if len(words) == 200:
#                     break

#     verify_list(words)

#     # Write to file
#     output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "val_words.txt")

#     with open(output_path, 'w') as f:
#         for word in words:
#             f.write(word + '\n')

#     print(f"✓ Written to: {output_path}")

#     # Print category breakdown
#     print(f"\nCategory breakdown:")
#     print(f"  Minimal pairs:       ~40 words (20 pairs)")
#     print(f"  Rhyming clusters:    ~40 words")
#     print(f"  Compound confusions: ~30 words (15 pairs)")
#     print(f"  Phonetically dense:  ~40 words")
#     print(f"  Borrowed words:      ~20 words")
#     print(f"  Anchor words:        ~30 words")
#     print(f"  (after dedup: {len(words)} total)")

# if __name__ == "__main__":
#     main()