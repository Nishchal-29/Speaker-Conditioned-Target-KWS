import urllib.request
import random

TARGET_SIZE = 3000

TECH_WORDS = {
    "python","docker","linux","github","gitlab",
    "cuda","tensorflow","pytorch","compiler",
    "kernel","postgres","redis","mongodb",
    "javascript","typescript","react","nodejs",
    "network","database","api","server","client",
    "frontend","backend","microservice","cloud"
}

ACRONYMS = {
    "cpu","gpu","ram","ssd","api","json","xml",
    "tcp","udp","dns","iit","mit","usb",
    "ssh","vpn","sql","html","css"
}

WAKE_WORDS = {
    "jarvis","alexa","siri","copilot","assistant",
    "atlas","nova","echo","apollo","aurora",
    "buddy","computer","oracle"
}

NAMES = {
    # Indian Male
    "rahul","rohan","arjun","akash","amit","vikram","sanjay",
    "nishchal","aditya","ayush","karan","manish","vivek","rohit",
    "sachin","abhishek","ankit","pranav","varun","harsh","yash",
    "nitin","deepak","gaurav","shubham","raj","sumit","krishna",

    # Indian Female
    "priya","ananya","neha","meera","pooja","kavya","isha",
    "shruti","riya","aarti","sneha","divya","nikita","payal",
    "anjali","kriti","shreya","tanya","simran","sonam",

    # Western Male
    "alex","john","david","james","michael","robert","daniel",
    "william","joseph","thomas","charles","henry","jack",
    "oliver","liam","noah","ethan","lucas","logan","benjamin",

    # Western Female
    "emma","sophia","olivia","ava","isabella","mia","amelia",
    "charlotte","harper","evelyn","abigail","ella","scarlett",
    "grace","victoria","zoe","lily","hannah","sarah","maria",

    # Mixed Global
    "sam","jordan","taylor","casey","morgan","jamie","alexis",
    "kai","leo","sofia","elena","ivan","diego","mateo"
}

LOCATIONS = {
    "mumbai","delhi","kolkata","bangalore",
    "hyderabad","chennai","london","paris",
    "berlin","tokyo","sydney","singapore",
    "dubai","seoul","newyork"
}

COMMANDS = {
    "start","stop","pause","resume","record",
    "cancel","search","save","delete","open",
    "close","upload","download","connect",
    "disconnect","launch","shutdown","restart",
    "mute","unmute"
}

HARD_NEGATIVES = {
    "start","smart","dart","part","tart",
    "stop","slop","bop","mop","cop","hop",
    "play","flay","slay","clay","dray","bray",
    "pause","cause","laws","paws","jaws",
    "resume","presume","consume","perfume",
    "next","text","hexed","vexed","flexed",
    "turn","burn","churn","earn","fern",
    "lights","brights","kites","fights","tights",
    "mute","cute","flute","lute","boot",
    "open","token","broken","spoken",
    "close","rose","hose","nose","pose",
    "timer","primer","rhymer","climber",
    "alarm","charm","harm","farm","swarm",
    "weather","feather","tether","heather","leather",
    "news","pews","dues","queues","views","muse"
}

def generate_nonwords(count=300):
    onsets = ["b","c","d","f","g","h","j","k","l","m","n","p","r","s","t","v","w","z",
              "br","cr","dr","fr","gr","pr","tr","kr","pl","cl","gl","bl","fl","sk","sp","st","sl"]
    vowels = ["a","e","i","o","u","ai","ee","oa","oo","ou"]
    codas = ["b","d","g","k","l","m","n","p","r","s","t","x","z","nd","nt","nk","mp","rd","rt"]
    custom = {"zorbin","talven","mirex","lodar","pralix","vintera","karven","dorex","meltra","gralon"}
    
    words = set(custom)
    while len(words) < count:
        word = random.choice(onsets) + random.choice(vowels) + random.choice(codas) + random.choice(vowels) + random.choice(codas)
        if 5 <= len(word) <= 8:
            words.add(word)
    return words

def is_redundant_inflection(word, seen_roots):
    if len(word) < 4: return False
    if word.endswith('s') and word[:-1] in seen_roots: return True
    if word.endswith('es') and word[:-2] in seen_roots: return True
    if word.endswith('ed') and word[:-2] in seen_roots: return True
    if word.endswith('ing') and word[:-3] in seen_roots: return True
    if word.endswith('ing') and word[:-3] + 'e' in seen_roots: return True
    if word.endswith('ed') and word[:-1] in seen_roots: return True 
    return False

def build_dictionary():
    core_vocab = set()
    for bucket in (TECH_WORDS, ACRONYMS, WAKE_WORDS, NAMES, LOCATIONS, COMMANDS, HARD_NEGATIVES):
        core_vocab.update(bucket)
    core_vocab.update(generate_nonwords(600))
    
    print(f"Loaded {len(core_vocab)} core priority words.")
    url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
    print("Downloading frequency list...")
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    lines = urllib.request.urlopen(req).read().decode("utf-8").splitlines()

    blocklist = {"hmm","uhh","umm","gonna","wanna","ain","cos","fuck","shit","bitch", "motherfucker","asshole","lmao","lol","rofl"}
    filler_pool = []
    seen_roots = set(core_vocab)

    for line in lines[:15000]:
        word = line.split()[0].lower()

        if not word.isalpha() or len(word) < 3 or len(word) > 12:
            continue
        if word in blocklist or word in core_vocab:
            continue
        if is_redundant_inflection(word, seen_roots):
            continue
            
        filler_pool.append(word)
        seen_roots.add(word)

    random.seed(42)
    random.shuffle(filler_pool)
    slots_needed = TARGET_SIZE - len(core_vocab)
    if slots_needed > 0:
        final_vocab = list(core_vocab) + filler_pool[:slots_needed]
    else:
        final_vocab = list(core_vocab)[:TARGET_SIZE]

    random.shuffle(final_vocab)
    with open("master_words.txt","w") as f:
        for w in final_vocab:
            f.write(w + "\n")

    print(f"Created master_words.txt with exactly {len(final_vocab)} words.")

if __name__ == "__main__":
    build_dictionary()