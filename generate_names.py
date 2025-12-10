import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import numpy as np
from datetime import datetime

# --- Configuration ---
WEIGHT_MEANING = 5.0
WEIGHT_GENDER = .1
WEIGHT_AGE = 3.0
WEIGHT_FREQUENCY = .5

AGE_BUCKETS = ['20-', '20~30', '30~40', '40~50', '50~60', '60+']
AGE_WEIGHTS = [6, 5, 4, 3, 2, 1]

def load_data(csv_path):
    """
    Load character data from given_names.csv.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    # Ensure character column is string
    df['character'] = df['character'].astype(str)
    return df

def get_embeddings(model, chars, cache_file='marathon_embeddings.npy'):
    """
    Compute or load embeddings for characters.
    """
    if os.path.exists(cache_file):
        print(f"Loading embeddings from {cache_file}...")
        embeddings = np.load(cache_file)
        if len(embeddings) != len(chars):
            print("Embedding count mismatch. Recomputing...")
            # Fall through to recompute
        else:
            return embeddings

    print("Computing embeddings (this may take a while)...")
    embeddings = model.encode(chars, convert_to_tensor=True, show_progress_bar=True)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    np.save(cache_file, embeddings)
    print(f"Embeddings saved to {cache_file}")
    return embeddings

def calculate_meaning_score(model, description, char_embeddings):
    """
    Calculate Cosine Similarity between description and characters.
    Returns: vector of scores.
    """
    desc_embedding = model.encode(description, convert_to_tensor=True)
    if isinstance(char_embeddings, np.ndarray):
        char_embeddings_tensor = torch.from_numpy(char_embeddings)
    else:
        char_embeddings_tensor = char_embeddings
        
    cos_scores = util.cos_sim(desc_embedding, char_embeddings_tensor)[0]
    return cos_scores.cpu().numpy()

def calculate_gender_score(df, user_gender):
    """
    Calculate gender score: log10(sex_ratio).
    Range [-3, 3].
    """
    # handle 999.0 (inf) and 0.0
    ratios = df['sex ratio(male/female)'].values.copy()
    
    # Clip large values to 1000 (log10 = 3)
    ratios[ratios > 999] = 1000.0
    # Clip small/zero values to 0.001 (log10 = -3)
    ratios[ratios < 0.001] = 0.001
    
    scores = np.log10(ratios)
    
    # Weight based on user gender
    # User Male: want High Ratio -> positive weight
    # User Female: want Low Ratio -> negative weight
    # Because log10(Ratio) is High for Male, Low (Negative) for Female names.
    # If Female, we want Low Score (-3) to become positive contribution (+3).
    # So multiply by -1.
    
    weight = 1.0 if user_gender == '男' else -1.0
    return scores * weight # This applies the direction

def calculate_age_score(df, birth_date):
    """
    Calculate weighted age score based on distance to usage frequency.
    """
    try:
        if isinstance(birth_date, str):
            dt = datetime.strptime(birth_date, "%Y-%m-%d")
        else:
            dt = birth_date
        
        current_year = datetime.now().year
        age = current_year - dt.year
    except:
        print("Invalid birth_date, approximating age score 0")
        return np.zeros(len(df))

    # Determine user's age bucket index
    # buckets: '20-' (<20), '20~30' (20-29), '30~40', '40~50', '50~60', '60+' (>=60)
    if age < 20: user_idx = 0
    elif age < 30: user_idx = 1
    elif age < 40: user_idx = 2
    elif age < 50: user_idx = 3
    elif age < 60: user_idx = 4
    else: user_idx = 5
    
    scores = np.zeros(len(df))
    
    # Calculate totals for each bucket (Male + Female columns)
    # The prompt says: "appearance count(both gender) of the character divided by the total number of athletes in that age range"
    # We approximate "Total number of athletes" by summing all character counts in that bucket across all rows.
    # (Assuming uniform name length distribution, normalizing by Sum(Counts) is equivalent to normalizing by Total Athletes for ranking purposes)
    
    bucket_total_counts = {}
    for b in AGE_BUCKETS:
        # Sum of 'male b' + 'female b' for ALL characters
        total_in_bucket = df[f'male {b}'].sum() + df[f'female {b}'].sum()
        bucket_total_counts[b] = total_in_bucket if total_in_bucket > 0 else 1.0

    for i, bucket in enumerate(AGE_BUCKETS):
        # Distance weight
        dist = abs(user_idx - i)
        if dist < len(AGE_WEIGHTS):
            weight = AGE_WEIGHTS[dist]
        else:
            weight = 0 # Should not happen with 6 buckets
            
        # Character count in this bucket (Male + Female)
        char_counts = df[f'male {bucket}'] + df[f'female {bucket}']
        
        # Normalized frequency
        norm_freq = char_counts / bucket_total_counts[bucket]
        
        scores += weight * norm_freq
        
    return scores

def calculate_frequency_score(df):
    """
    log10(total appearance count).
    """
    total_counts = df['total male'] + df['total female']
    # Handle 0 or 1
    total_counts[total_counts < 1] = 1
    return np.log10(total_counts)

def load_family_names(csv_path):
    df = pd.read_csv(csv_path)
    return df

def sample_family_name(family_df):
    ranks = family_df['rank'].values
    weights = 1.0 / (10.0 + ranks)
    probs = weights / np.sum(weights)
    selected_idx = np.random.choice(len(family_df), p=probs)
    return family_df.iloc[selected_idx]['family name']

def main():
    # --- Data Paths ---
    char_csv_path = 'given_names.csv'
    family_csv_path = 'family_name.csv'
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    
    # --- User Params (Example) ---
    user_params = {
        "english_name": "Robert",
        "gender": "男",
        "description": "美麗",
        "birth_date": "1980-05-20"
    }
    
    print("Loading data...")
    try:
        char_df = load_data(char_csv_path)
        family_df = load_family_names(family_csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Loading model...")
    model = SentenceTransformer(model_name)
    
    chars = char_df['character'].tolist()
    char_embeddings = get_embeddings(model, chars)
    
    # --- Calculate Scores ---
    print("Calculating scores...")
    
    # 1. Meaning Score
    s_meaning = calculate_meaning_score(model, user_params['description'], char_embeddings)
    s_meaning_weighted = s_meaning * WEIGHT_MEANING
    # 2. Gender Score
    # Note: user_params['gender'] determines direction
    s_gender = calculate_gender_score(char_df, user_params['gender'])
    s_gender_weighted = s_gender * WEIGHT_GENDER 
    # s_gender_weighted already includes the direction flip for females if needed, 
    # but based on prompt "w_g * gender_score". 
    # My function returns (score * weight).
    # If we stick strictly to prompt: gender_score is log10(ratio). w_g is set to 1.
    # But for Female user, we need negative log10 to be good.
    # So I effectively implicitly set w_g = -1 inside the function for Female.
    
    # 3. Age Score
    s_age = calculate_age_score(char_df, user_params['birth_date'])
    s_age_weighted = s_age * WEIGHT_AGE
    
    # 4. Frequency Score
    s_freq = calculate_frequency_score(char_df)
    s_freq_weighted = s_freq * WEIGHT_FREQUENCY
    
    # --- Total Score ---
    # Normalize scores? The prompt implies direct sum.
    # Meaning: [-1, 1] usually [0, 0.5] for cross-lingual.
    # Gender: [-3, 3]
    # Frequency: [0, ~4] (log10(10000))
    # Age: weighted sum of small probabilities. 
    #   Prob ~ 1/1000 = 0.001. Sum ~ 6*0.001 = 0.006. 
    #   Age score might be very small compared to others.
    #   Prompt didn't ask to normalize, but "w_a=1" might make it insignificant.
    #   "age_interval_weight = [6,5...]" -> max 21. 
    #   If prop is 0.01 -> score 0.2. 
    #   Maybe it's fine.
    
    total_score = (s_meaning_weighted + 
                   s_gender_weighted + 
                   s_age_weighted + 
                   s_freq_weighted)
                   
    # Add score to DF for visualization/debugging
    char_df['score'] = total_score
    char_df['s_meaning'] = s_meaning_weighted
    char_df['s_gender'] = s_gender_weighted
    char_df['s_age'] = s_age_weighted
    char_df['s_frequency'] = s_freq_weighted
    
    # Sort by score
    top_df = char_df.sort_values(by='score', ascending=False)
    
    print("-" * 30)
    print(f"Top 5 Characters for {user_params['english_name']} ({user_params['gender']}):")
    print(top_df[['character', 'score', 's_meaning', 's_gender', 's_age', 's_frequency']].head(30))
    print("-" * 30)
    
    # --- Generate Names ---
    print(f"Generating 30 names for meaning: '{user_params['description']}'")
    
    # Sampling Strategy:
    # Use softmax or simple weighted probability on top K?
    # Original used weighted similarity on Top 100.
    # New score can be negative. Softmax is safer.
    # Or shift to positive.
    
    top_k = 100
    candidates = top_df.head(top_k)
    cand_chars = candidates['character'].values
    
    # Softmax on scores
    scores = candidates['score'].values
    exp_scores = np.exp(scores - np.max(scores) + 15) # shift for stability
    probs = exp_scores / np.sum(exp_scores)
    
    generated_names = set()
    count = 0
    attempts = 0
    
    while count < 30 and attempts < 1000:
        attempts += 1
        fam = sample_family_name(family_df)
        
        # Pick 2 chars (with replacement allowed by user logic, but usually we filter dupes if unwanted)
        # prompt: "Probabilistically sample two given name characters"
        # We can pick 2 independently from the distribution
        
        given_chars = np.random.choice(cand_chars, size=2, p=probs, replace=False)
        full_name = fam + "".join(given_chars)
        
        if full_name not in generated_names:
            generated_names.add(full_name)
            print(f"{count+1:2d}. {full_name}")
            count += 1

if __name__ == '__main__':
    main()
