from phonemizer import phonemize
from pypinyin import lazy_pinyin, pinyin, Style
import epitran
import panphon.featuretable as ft
import re
import symbols

table = ft.FeatureTable()

def add_aspiration(ipa_text):
    """
    Adds aspiration to voiceless stops (p, t, k) when they appear
    at the beginning of a word or the beginning of a stressed syllable.
    """
    # Add aspiration at word start
    ipa_text = re.sub(r'\b([ptk])', r'\1ʰ', ipa_text)
    
    # Add aspiration after primary stress mark ˈ
    ipa_text = re.sub(r'ˈ([ptk])', r'ˈ\1ʰ', ipa_text)
    
    return ipa_text

def tokenize_ipa(ipa_text):
    """
    Tokenize IPA string into a list of valid symbols from symbols.py.
    Greedy match against sorted symbols (longest first).
    """
    tokens = []
    # Combine all symbols and sort by length descending to match longest first (e.g. tʃ before t)
    all_syms = sorted(list(set(symbols.cons_s + symbols.vowels + symbols.cons_e)), key=len, reverse=True)
    
    i = 0
    while i < len(ipa_text):
        match = False
        for sym in all_syms:
            if ipa_text.startswith(sym, i):
                tokens.append(sym)
                i += len(sym)
                match = True
                break
        if not match:
            # Skip unknown character or treat as individual token
            tokens.append(ipa_text[i])
            i += 1
    return tokens

def get_english_syllables(word):
    """
    Transcribe English word to IPA, add aspiration, and segment into Chinese-like syllables.
    Structure: C + V(optional) + C(optional)
    """
    # Note: phonemizer requires 'espeak' backend installed on the system.
    ipa = phonemize(
        word,
        language='en-us',
        backend='espeak',
        strip=True,
        with_stress=True
    )
    
    # Add aspiration
    ipa = add_aspiration(ipa)
    
    # Remove stress markers for processing
    ipa_clean = ipa.replace('ˈ', '').replace('ˌ', '').replace('ː', '')
    
    tokens = tokenize_ipa(ipa_clean)
    syllables = []
    current_syllable = []
    
    for i,token in enumerate(tokens):
        if not current_syllable:
            current_syllable.append(token)
            continue

        is_v = token in symbols.vowels
        is_c_s = token in symbols.cons_s
        is_c_e = token in symbols.cons_e
        has_v = any(t in symbols.vowels for t in current_syllable)
        last_is_v = current_syllable[-1] in symbols.vowels
        
        # vowel
        if is_v and not has_v:
            current_syllable.append(token)
        # ending consonant
        elif last_is_v and is_c_e and (i+1 >= len(tokens) or tokens[i+1] not in symbols.vowels):
            current_syllable.append(token)
        else:
            syllables.append("".join(current_syllable))
            current_syllable = [token]
                
    if current_syllable:
        syllables.append("".join(current_syllable))

    return syllables

def get_chinese_syllables(text):
    """
    Convert Traditional Chinese to Pinyin, then to IPA syllables.
    """
    pinyins = lazy_pinyin(text)
    ipa_syllables = []
    epi = epitran.Epitran('cmn-Latn')
    
    for p in pinyins:
        ipa = epi.transliterate(p).replace(" ", "")
        ipa_syllables.append(ipa)
        
    return ipa_syllables

def decompose_syllable(syllable_str):
    """
    Decompose a syllable string into (Onset, Nucleus, Coda).
    Returns a tuple of strings (onset, nucleus, coda).
    Each part is None if missing.
    Assumes the syllable was formed using the C+V+C logic.
    """
    tokens = tokenize_ipa(syllable_str)
    
    onset = []
    nucleus = []
    coda = []
    
    # Simple state machine to parse back the CVC structure
    # Since we built it as C(opt) + V(opt) + C(opt), we can scan for V.
    
    # Find the first vowel (Nucleus)
    v_index = -1
    for i, token in enumerate(tokens):
        if token in symbols.vowels:
            v_index = i
            break
            
    if v_index != -1:
        # Found Nucleus
        nucleus = [tokens[v_index]]
        # Everything before is Onset
        onset = tokens[:v_index]
        # Everything after is Coda
        coda = tokens[v_index+1:]
    else:
        # No Nucleus (Vowel). Onset only.
        onset = tokens
        nucleus = []
        coda = []

    def to_str(token_list):
        if not token_list:
            return None
        return token_list[0]

    return to_str(onset), to_str(nucleus), to_str(coda)

def compute_part_distance(seg1, seg2):
    """
    Compute distance between two segments.
    If both None: 0
    If one None: 0.25
    Else: feature distance
    """
    if seg1 is None and seg2 is None:
        return 0.0
    if seg1 is None or seg2 is None:
        return 0.25
        
    # Handle 'ɚ' manually
    s1 = 'ə' if seg1 == 'ɚ' else seg1
    s2 = 'ə' if seg2 == 'ɚ' else seg2
    try:
        vec1 = table.word_to_vector_list(s1, numeric=True)[0]
        vec2 = table.word_to_vector_list(s2, numeric=True)[0]
        return sum(abs(v1 - v2) for v1, v2 in zip(vec1, vec2)) / 24.0
    except Exception:
        # Fallback if symbol not found
        return 1.0

def compute_syllable_distance(syl1, syl2):
    """
    Compute distance between two IPA syllables by decomposing into C+V+C
    and summing partial distances.
    """
    o1, n1, c1 = decompose_syllable(syl1)
    o2, n2, c2 = decompose_syllable(syl2)
    
    d_onset = compute_part_distance(o1, o2)
    d_nucleus = compute_part_distance(n1, n2)
    d_coda = compute_part_distance(c1, c2)
    dist = d_onset + d_nucleus + d_coda
    return dist

def compute_name_distance(en_syllables, zh_syllables):
    """
    Compute edit distance between English name and Chinese name
    using syllables as units.
    """
    n = len(en_syllables)
    m = len(zh_syllables)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    GAP_COST = 1.0 
    
    for i in range(1, n + 1):
        dp[i][0] = i * GAP_COST
    for j in range(1, m + 1):
        dp[0][j] = j * GAP_COST
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = compute_syllable_distance(en_syllables[i-1], zh_syllables[j-1])
            # print(cost_sub,en_syllables[i],zh_syllables[i])
            dp[i][j] = min(
                dp[i-1][j] + GAP_COST,      # Deletion of English syllable
                dp[i][j-1] + GAP_COST,      # Insertion of Chinese syllable
                dp[i-1][j-1] + cost_sub     # Substitution
            )
            
    # Backtracking
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost_sub = compute_syllable_distance(en_syllables[i-1], zh_syllables[j-1])
            # Check if substitution
            # Use small epsilon for float comparison if needed, or just direct comparison
            if abs(dp[i][j] - (dp[i-1][j-1] + cost_sub)) < 1e-9:
                alignment.append((en_syllables[i-1], zh_syllables[j-1], cost_sub))
                i -= 1
                j -= 1
                continue
        
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP_COST)) < 1e-9:
            # Deletion (English syllable has no match)
            alignment.append((en_syllables[i-1], "", GAP_COST))
            i -= 1
        elif j > 0 and abs(dp[i][j] - (dp[i][j-1] + GAP_COST)) < 1e-9:
            # Insertion (Chinese syllable has no match)
            alignment.append(("", zh_syllables[j-1], GAP_COST))
            j -= 1
        else:
            # Should not happen if DP is correct
            break
            
    alignment.reverse()
            
    return dp[n][m], alignment

def compute_tone_score(word):
    tone_numbers = [int(p[0][-1]) if p[0][-1].isdigit() else 5 for p in pinyin(word, style=Style.TONE3)]
    print(tone_numbers)
    return

if __name__ == '__main__':
    # english_name = "Jason"
    # chinese_name = "傑森"
    # compute_tone_score(chinese_name)
    # en_syls = get_english_syllables(english_name)
    # print(f"English: {english_name} -> {en_syls}")
    # zh_syls = get_chinese_syllables(chinese_name)
    # print(f"Chinese: {chinese_name} -> {zh_syls}")
    # dist, alignment = compute_name_distance(en_syls, zh_syls)
    # print(f"Phonetic distance: {dist:.2f}")
    # exit()
    english_name = 'robert'
    chinese_names = [
    "羅伯特",
    "羅伯傑",
    "林百哲",
]

    en_syls = get_english_syllables(english_name)
    print(f"English: {english_name} -> {en_syls}")
    for chinese_name in chinese_names:
        zh_syls = get_chinese_syllables(chinese_name)
        print(f"Chinese: {chinese_name} -> {zh_syls}")
        dist, alignment = compute_name_distance(en_syls, zh_syls)
        print(f"Phonetic distance: {dist:.2f}")
        for en, zh, cost in alignment:
            print(f"('{en}', '{zh}') -> {cost:.2f}")
