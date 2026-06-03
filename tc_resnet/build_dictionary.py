import urllib.request
import random

def build_master_wordlist(target_count=30000, min_len=3, max_len=12, output_file="master_words.txt"):
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"    
    response = urllib.request.urlopen(url)
    raw_words = response.read().decode('utf-8').splitlines()    
    valid_words = [w.lower() for w in raw_words if w.isalpha() and min_len <= len(w) <= max_len]
    
    valid_words = list(set(valid_words))
    print(f"Filtered to {len(valid_words):,} valid words (length {min_len}-{max_len}).")

    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(valid_words)
    final_words = valid_words[:target_count]
    final_words.sort() 
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in final_words:
            f.write(f"{word}\n")
            
    print(f"Successfully wrote {len(final_words):,} words to {output_file}")

if __name__ == "__main__":
    build_master_wordlist(target_count=30000)