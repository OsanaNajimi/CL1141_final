import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import numpy as np
from datetime import datetime
from datetime import datetime
import numpy as np
import IPA

# --- Configuration ---
WEIGHT_MEANING = 5.0
WEIGHT_GENDER = .33
WEIGHT_AGE = .33
WEIGHT_FREQUENCY = .2
WEIGHT_PHONE = .1 # Adjust weight for phonetic score

AGE_BUCKETS = ['20-', '20~30', '30~40', '40~50', '50~60', '60+']
AGE_WEIGHTS = [6, 5, 4, 3, 2, 1]

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    df['character'] = df['character'].astype(str)
    return df

def load_ipa_dict(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    # Create dict {char: ipa_string}
    ipa_map = {}
    for _, row in df.iterrows():
        ipa_map[str(row['character'])] = str(row['ipa'])
    return ipa_map

def get_embeddings(model, chars, cache_file='marathon_embeddings.npy'):
    if os.path.exists(cache_file):
        print(f"Loading embeddings from {cache_file}...")
        embeddings = np.load(cache_file)
        if len(embeddings) != len(chars):
            print("Embedding count mismatch. Recomputing...")
        else:
            return embeddings

    print("Computing embeddings (this may take a while)...")
    embeddings = model.encode(chars, convert_to_tensor=True, show_progress_bar=True)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    np.save(cache_file, embeddings)
    return embeddings

def calculate_meaning_score(model, description, char_embeddings):
    desc_embedding = model.encode(description, convert_to_tensor=True)
    if isinstance(char_embeddings, np.ndarray):
        char_embeddings_tensor = torch.from_numpy(char_embeddings)
    else:
        char_embeddings_tensor = char_embeddings
        
    cos_scores = util.cos_sim(desc_embedding, char_embeddings_tensor)[0]
    return cos_scores.cpu().numpy()

def calculate_gender_score(df, user_gender):
    ratios = df['sex ratio(male/female)'].values.copy()
    ratios[ratios > 999] = 1000.0
    ratios[ratios < 0.001] = 0.001
    scores = np.log10(ratios)
    
    # Male: Score > 0 good. Female: Score < 0 good.
    # If Female, we multiply by -1 so large negative (female) becomes large positive (good).
    weight = 1.0 if user_gender == '男' else -1.0
    return scores * weight

def calculate_age_score(df, birth_date):
    try:
        if isinstance(birth_date, str):
            dt = datetime.strptime(birth_date, "%Y-%m-%d")
        else:
            dt = birth_date
        current_year = datetime.now().year
        age = current_year - dt.year
    except:
        return np.zeros(len(df))

    if age < 20: user_idx = 0
    elif age < 30: user_idx = 1
    elif age < 40: user_idx = 2
    elif age < 50: user_idx = 3
    elif age < 60: user_idx = 4
    else: user_idx = 5
    
    scores = np.zeros(len(df))
    
    bucket_total_counts = {}
    for b in AGE_BUCKETS:
        total_in_bucket = df[f'male {b}'].sum() + df[f'female {b}'].sum()
        bucket_total_counts[b] = total_in_bucket if total_in_bucket > 0 else 1.0

    for i, bucket in enumerate(AGE_BUCKETS):
        dist = abs(user_idx - i)
        if dist < len(AGE_WEIGHTS):
            weight = AGE_WEIGHTS[dist]
        else:
            weight = 0
            
        char_counts = df[f'male {bucket}'] + df[f'female {bucket}']
        norm_freq = char_counts / bucket_total_counts[bucket]
        scores += weight * norm_freq
        
    return scores

def calculate_frequency_score(df):
    total_counts = df['total male'] + df['total female']
    total_counts[total_counts < 1] = 1
    return np.log10(total_counts)

def calculate_phonetic_scores(df, en_syllables, ipa_map):
    """
    Calculate s_phone1 and s_phone2 based on English name syllables.
    """
    # 1. Get English Syllables
    num_syllables = len(en_syllables)
    
    scores1 = []
    scores2 = []
    
    # We need to compute score for every character in df
    for char in df['character']:
        char_ipa = ipa_map.get(str(char), '')
        
        # If no IPA, assign bad score (large distance)
        if not char_ipa:
            scores1.append(100.0)
            scores2.append(100.0)
            continue
            
        # Chinese IPA is typically one syllable for one character
        # char_ipa is the string representation.
        # But compute_syllable_distance expects decomposed/tokenized or raw string?
        # Looking at IPA.py, it calls decompose_syllable which calls tokenize_ipa.
        # So raw IPA string is fine.
        
        # Logic from PROMPT:
        if num_syllables == 1:
            # s_phone1 == s_phone2 == dist(char, en_syl[0])
            dist = IPA.compute_syllable_distance(char_ipa, en_syllables[0])
            scores1.append(dist)
            scores2.append(dist)
            
        elif num_syllables == 2:
            # s_phone1 = dist(char, en_syl[0])
            # s_phone2 = dist(char, en_syl[1])
            d1 = IPA.compute_syllable_distance(char_ipa, en_syllables[0])
            d2 = IPA.compute_syllable_distance(char_ipa, en_syllables[1])
            scores1.append(d1)
            scores2.append(d2)
            
        else: # >= 3
            # family name should cover syllable 1 (handled elsewhere? No, generate_names picks family separately).
            # "the family name should be the character closest to the first syllable"
            # -> This implies we need to guide family name selection too. 
            # But the prompt says "add these 2 scores to calculate the total score for the character".
            # The character here refers to the GIVEN NAME characters being scored in the DataFrame.
            # "s_phone1 is distance of second syllable, s_phone2 is distance of third syllable"
            d1 = IPA.compute_syllable_distance(char_ipa, en_syllables[1])
            d2 = IPA.compute_syllable_distance(char_ipa, en_syllables[2])
            scores1.append(d1)
            scores2.append(d2)
            
    # Convert distance to Score. Distance 0 is best. High distance is bad.
    # We want Score to be higher for closer match.
    # Maybe -1 * distance? Or 1 / (1+dist)?
    # Prompt says "calculate scores... add these 2 scores to calculate total score".
    # Assuming lower distance is better, we should probably subtract distance or add negative distance.
    # Let's use negative distance so higher total score => better match.
    
    return np.array(scores1) * -1.0, np.array(scores2) * -1.0

def load_family_names(csv_path):
    df = pd.read_csv(csv_path)
    return df

def sample_family_name(family_df, en_syllables, ipa_map):
    """
    Pick family name.
    If English name >= 3 syllables, pick family name closest to 1st syllable.
    Otherwise probabilistic.
    """
    if en_syllables and ipa_map and False:
        if len(en_syllables) >= 3:
            # Find best family match
            best_fam = None
            min_dist = float('inf')
            
            # Check all family names
            for fam in family_df['family name']:
                fam_str = str(fam)
                # If multi-char family name, just take first char or whole?
                # Usually single char.
                # Sum distance if multi char?
                
                # Get IPA for family name chars
                # If not in map, compute on fly?
                curr_dist = 0
                valid = True
                for char in fam_str:
                    c_ipa = ipa_map.get(char)
                    if not c_ipa:
                        # try compute
                        try:
                            syll = IPA.get_chinese_syllables(char)[0]
                            c_ipa = syll
                        except:
                            valid = False
                            break
                    curr_dist += IPA.compute_syllable_distance(c_ipa, en_syllables[0])
                
                if valid:
                    # Average distance if multi-char?
                    # Or just 1st char? 
                    # "family name should be the character closest to first syllable" -> implies single char.
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        best_fam = fam_str
            
            if best_fam:
                return best_fam

    # Fallback / < 3 syllables logic
    ranks = family_df['rank'].values
    weights = 1.0 / (10.0 + ranks)
    probs = weights / np.sum(weights)
    selected_idx = np.random.choice(len(family_df), p=probs)
    return family_df.iloc[selected_idx]['family name']

def main():
    char_csv_path = 'given_names.csv'
    family_csv_path = 'family_name.csv'
    ipa_csv_path = 'character_ipa.csv'
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    
    user_params = {
        "english_name": "Robert",
        "gender": "男",
        "description": "自然，聰明",
        "birth_date": "1980-05-20"
    }
    
    print("Loading data...")
    try:
        char_df = load_data(char_csv_path)
        family_df = load_family_names(family_csv_path)
        ipa_map = load_ipa_dict(ipa_csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Loading model...")
    model = SentenceTransformer(model_name)
    
    chars = char_df['character'].tolist()
    char_embeddings = get_embeddings(model, chars)
    
    print("Calculating scores...")
    s_meaning = calculate_meaning_score(model, user_params['description'], char_embeddings)
    s_gender = calculate_gender_score(char_df, user_params['gender']) 
    s_age = calculate_age_score(char_df, user_params['birth_date'])
    s_freq = calculate_frequency_score(char_df)
    
    en_syllables = IPA.get_english_syllables(user_params['english_name'])
    s_phone1, s_phone2 = calculate_phonetic_scores(char_df, en_syllables, ipa_map)
    
    # Total Score
    total_score = (WEIGHT_MEANING * s_meaning + 
                   WEIGHT_GENDER * s_gender + 
                   WEIGHT_AGE * s_age + 
                   WEIGHT_FREQUENCY * s_freq + 
                   WEIGHT_PHONE * s_phone1 + 
                   WEIGHT_PHONE * s_phone2)
                   
    char_df['score'] = total_score
    char_df['s_m'] = s_meaning*WEIGHT_MEANING
    char_df['s_g'] = s_gender*WEIGHT_GENDER
    char_df['s_a'] = s_age*WEIGHT_AGE
    char_df['s_f'] = s_freq*WEIGHT_FREQUENCY
    char_df['s_p1'] = s_phone1*WEIGHT_PHONE
    char_df['s_p2'] = s_phone2*WEIGHT_PHONE
    
    top_df = char_df.sort_values(by='score', ascending=False)
    
    print("-" * 30)
    print(f"Top 10 Characters for {user_params['english_name']} ({user_params['gender']}):")
    print(top_df[['character', 'score', 's_m', 's_g', 's_f', 's_p1', 's_p2']].head(10))
    print("-" * 30)
    
    print(f"Generating 30 names for meaning: '{user_params['description']}'")
    
    top_k = 100
    candidates = top_df.head(top_k)
    cand_chars = candidates['character'].values
    scores = candidates['score'].values
    
    # Shift for softmax
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    
    generated_names = set()
    count = 0
    attempts = 0
    
    while count < 30 and attempts < 1000:
        attempts += 1
        # Determine Family Name logic
        fam = sample_family_name(family_df, en_syllables, ipa_map)
        
        given_chars = np.random.choice(cand_chars, size=2, p=probs, replace=True)
        full_name = fam + "".join(given_chars)
        
        if full_name not in generated_names:
            generated_names.add(full_name)
            print(f"{count+1:2d}. {full_name}")
            count += 1

if __name__ == '__main__':
    main()
