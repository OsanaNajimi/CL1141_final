import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import numpy as np
from datetime import datetime
import IPA

# --- Configuration Constants ---
WEIGHT_MEANING = 5.0
WEIGHT_GENDER = 0.33
WEIGHT_AGE = 0.33
WEIGHT_PHONE = 10.0 
TEMPERATURE = 5.0 # temperature for generation

AGE_BUCKETS = ['20-', '20~30', '30~40', '40~50', '50~60', '60+']
AGE_WEIGHTS = [6, 5, 4, 3, 2, 1]

class NameGenerator:
    def __init__(self, char_csv_path='given_names.csv', 
                 family_csv_path='family_name.csv', 
                 ipa_csv_path='character_ipa.csv',
                 meaning_csv_path='character_meaning.csv',
                 model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        
        print("NameGenerator: Loading data...")
        if not os.path.exists(char_csv_path):
            raise FileNotFoundError(f"{char_csv_path} not found.")
        self.char_df = pd.read_csv(char_csv_path)
        self.char_df['character'] = self.char_df['character'].astype(str)
        
        if not os.path.exists(family_csv_path):
             raise FileNotFoundError(f"{family_csv_path} not found.")
        self.family_df = pd.read_csv(family_csv_path)

        if not os.path.exists(ipa_csv_path):
            # Fallback handling or raise error? User code implies it exists.
            print(f"Warning: {ipa_csv_path} not found. Phonetic scores might be impaired if not in given_names.")
            self.ipa_map = {}
        else:
            self.ipa_map = self.load_ipa_dict(ipa_csv_path)

        if not os.path.exists(meaning_csv_path):
             print(f"Warning: {meaning_csv_path} not found. Character meanings will be missing.")
             self.meaning_map = {}
        else:
            self.meaning_map = self.load_meaning_dict(meaning_csv_path)
            
        print("NameGenerator: Loading model...")
        self.model = SentenceTransformer(model_name)
        
        # Precompute embeddings for characters
        print("NameGenerator: Precomputing character embeddings...")
        self.chars = self.char_df['character'].tolist()
        self.char_embeddings = self.get_embeddings(self.chars)
        print("NameGenerator: Ready.")

    def load_ipa_dict(self, csv_path):
        df = pd.read_csv(csv_path)
        ipa_map = {}
        for _, row in df.iterrows():
            ipa_map[str(row['character'])] = str(row['ipa'])
        return ipa_map

    def load_meaning_dict(self, csv_path):
        df = pd.read_csv(csv_path)
        meaning_map = {}
        for _, row in df.iterrows():
            meaning_map[str(row['Character'])] = str(row['Meaning'])
        return meaning_map

    def get_embeddings(self, chars, cache_file='marathon_embeddings.npy'):
        if os.path.exists(cache_file):
            # Verify length
            embeddings = np.load(cache_file)
            if len(embeddings) == len(chars):
                return embeddings
            print("Embedding cache mismatch, recomputing...")
        
        embeddings = self.model.encode(chars, convert_to_tensor=True, show_progress_bar=True)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        np.save(cache_file, embeddings)
        return embeddings

    def calculate_scores(self, user_params):
        """
        Calculate scores for all characters based on user params.
        """
        description = user_params.get('description', '')
        gender = user_params.get('gender', 'neutral')
        birth_date = user_params.get('birth_date', '2000-01-01')
        english_name = user_params.get('english_name', '')

        # 1. Meaning Score
        desc_embedding = self.model.encode(description, convert_to_tensor=True)
        # util.cos_sim returns tensor
        cos_scores = util.cos_sim(desc_embedding, self.char_embeddings)[0].cpu().numpy()
        s_meaning = cos_scores

        # 2. Gender Score
        s_gender = self.calculate_gender_score(self.char_df, gender)

        # 3. Age Score
        s_age = self.calculate_age_score(self.char_df, birth_date)

        # 4. Frequency Score (s_freq)
        s_freq = self.calculate_frequency_score(self.char_df)

        # 5. Phonetic Score
        en_syllables = IPA.get_english_syllables(english_name) if english_name else []
        s_phone1, s_phone2 = self.calculate_phonetic_scores(self.char_df, en_syllables, self.ipa_map)

        # Total Score Calculation (Preserving User Logic)
        # total_score = (Weights * Components) * s_freq
        # Note: s_freq is log10 of counts, so it acts as a multiplier based on popularity.
        
        total_term = (WEIGHT_MEANING * s_meaning + 
                      WEIGHT_GENDER * s_gender + 
                      WEIGHT_AGE * s_age + 
                      WEIGHT_PHONE * s_phone1 + 
                      WEIGHT_PHONE * s_phone2)
        
        total_score = total_term * s_freq
        
        # Create a result dataframe
        res_df = self.char_df.copy()
        res_df['score'] = total_score
        res_df['s_mean'] = s_meaning * WEIGHT_MEANING
        res_df['s_gend'] = s_gender * WEIGHT_GENDER
        res_df['s_age'] = s_age * WEIGHT_AGE
        res_df['s_freq'] = s_freq
        res_df['s_ph1'] = s_phone1 * WEIGHT_PHONE
        res_df['s_ph2'] = s_phone2 * WEIGHT_PHONE
        
        return res_df, en_syllables

    def generate_names(self, user_params, top_k=100, num_names=30):
        """
        Generate list of names.
        Input: user_params (dict)
        Output: list of dicts [{'name': '...', 'scores': {...}}]
        """
        df_scored, en_syllables = self.calculate_scores(user_params)
        
        # Filter top K
        top_df = df_scored.sort_values(by='score', ascending=False).head(top_k)
        
        # Probabilities for sampling
        scores = top_df['score'].values
        # Softmax with temperature
        exp_scores = np.exp(scores / TEMPERATURE)
        probs = exp_scores / np.sum(exp_scores)
        
        cand_chars = top_df['character'].values
        
        generated_results = []
        generated_names_set = set()
        attempts = 0
        
        while len(generated_results) < num_names and attempts < 1000:
            attempts += 1
            
            # Sample Family Name (Probabilistic or Phonetical)
            # Pass ipa_map and syllables for phonetic matching if applicable
            fam = self.sample_family_name(self.family_df, en_syllables, self.ipa_map)
            
            # Sample Given Name (2 chars)
            # Weighted random choice
            given_chars_indices = np.random.choice(len(cand_chars), size=2, p=probs, replace=True)
            g1 = cand_chars[given_chars_indices[0]]
            g2 = cand_chars[given_chars_indices[1]]
            
            # Get their partial scores
            row1 = top_df.iloc[given_chars_indices[0]]
            row2 = top_df.iloc[given_chars_indices[1]]
            
            full_name = fam + g1 + g2
            
            if full_name not in generated_names_set:
                generated_names_set.add(full_name)
                
                name_info = {
                    "name": full_name,
                    "family_name": fam,
                    "given_name": g1 + g2,
                    "total_score": float(row1['score'] + row2['score']), # Simple sum of individual scores
                    "details": {
                        "char1": {
                            "char": g1,
                            "meaning": float(row1['s_mean']),
                            "gender": float(row1['s_gend']),
                            "age": float(row1['s_age']),
                            "freq": float(row1['s_freq']),
                            "phone": float(row1['s_ph1'] + row1['s_ph2']) # Sum phone parts
                        },
                        "char2": {
                            "char": g2,
                            "meaning": float(row2['s_mean']),
                            "gender": float(row2['s_gend']),
                            "age": float(row2['s_age']),
                            "freq": float(row2['s_freq']),
                            "phone": float(row2['s_ph1'] + row2['s_ph2'])
                        }
                    }
                }
                generated_results.append(name_info)
        
        # Extract top 30 characters
        top_characters_list = []
        # top_df is already sorted by score descending
        top_30_chars = top_df.head(30)
 
        user_gender = user_params.get('gender', '男')
        gender_weight = 1.0 if user_gender == '男' else -1.0
        
        for _, row in top_30_chars.iterrows():
            # Calculate absolute gender score
            # s_gend in df is s_gender_raw * WEIGHT_GENDER
            # s_gender_raw = log10(ratio) * gender_weight
            # We want absolute_gender_score = log10(ratio) / 3
            # So abs_score = (s_gend / WEIGHT_GENDER / gender_weight) / 3
            
            s_gend_val = float(row['s_gend'])
            try:
                raw_gender = s_gend_val / WEIGHT_GENDER
                # log_ratio * gender_weight = raw_gender
                # log_ratio = raw_gender / gender_weight
                log_ratio = raw_gender / gender_weight
                abs_gender_score = log_ratio / 3.0
            except:
                abs_gender_score = 0.0

            char_str = str(row['character'])
            char_info = {
                "character": char_str,
                "total_score": float(row['score']),
                "meaning": self.meaning_map.get(char_str, ""),
                "absolute_gender_score": abs_gender_score,
                "details": {
                    "meaning": float(row['s_mean']),
                    "gender": float(row['s_gend']),
                    "age": float(row['s_age']),
                    "freq": float(row['s_freq']),
                    "phone": float(row['s_ph1'] + row['s_ph2'])
                }
            }
            top_characters_list.append(char_info)
        # print(top_characters_list)
        # print(generated_results)
        return {
            "recommendations": generated_results,
            "top_characters": top_characters_list
        }

    # --- Helper Calculation Methods (Preserving Logic) ---
    def calculate_gender_score(self, df, user_gender):
        ratios = df['sex ratio(male/female)'].values.copy()
        ratios[ratios > 999] = 1000.0
        ratios[ratios < 0.001] = 0.001
        scores = np.log10(ratios)
        weight = 1.0 if user_gender == 'male' else -1.0
        return scores * weight

    def calculate_age_score(self, df, birth_date):
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
            total = df[f'male {b}'].sum() + df[f'female {b}'].sum()
            bucket_total_counts[b] = total if total > 0 else 1.0

        for i, bucket in enumerate(AGE_BUCKETS):
            dist = abs(user_idx - i)
            weight = AGE_WEIGHTS[dist] if dist < len(AGE_WEIGHTS) else 0
            
            char_counts = df[f'male {bucket}'] + df[f'female {bucket}']
            norm_freq = char_counts / bucket_total_counts[bucket]
            scores += weight * norm_freq
        return scores

    def calculate_frequency_score(self, df):
        total_counts = df['total male'] + df['total female']
        total_counts[total_counts < 1] = 1
        return np.log10(total_counts)

    def calculate_phonetic_scores(self, df, en_syllables, ipa_map):
        num_syllables = len(en_syllables)
        scores1 = []
        scores2 = []
        
        for char in df['character']:
            char_ipa = ipa_map.get(str(char), '')
            if not char_ipa:
                scores1.append(0)
                scores2.append(0)
                continue
                
            threshold = 0.4
            
            if num_syllables == 1:
                d1 = IPA.compute_syllable_distance(char_ipa, en_syllables[0])
                s1 = (threshold - d1) / threshold if d1 < threshold else 0
                scores1.append(s1)
                scores2.append(s1)
                
            elif num_syllables == 2:
                d1 = IPA.compute_syllable_distance(char_ipa, en_syllables[0])
                d2 = IPA.compute_syllable_distance(char_ipa, en_syllables[1])
                s1 = (threshold - d1) / threshold if d1 < threshold else 0
                s2 = (threshold - d2) / threshold if d2 < threshold else 0
                scores1.append(s1)
                scores2.append(s2)
                
            else: # >= 3
                d1 = IPA.compute_syllable_distance(char_ipa, en_syllables[1])
                d2 = IPA.compute_syllable_distance(char_ipa, en_syllables[2])
                s1 = (threshold - d1) / threshold if d1 < threshold else 0
                s2 = (threshold - d2) / threshold if d2 < threshold else 0
                scores1.append(s1)
                scores2.append(s2)
                
        return np.array(scores1), np.array(scores2)

    def sample_family_name(self, family_df, en_syllables, ipa_map):
        # Fallback / < 3 syllables logic or always probabilistic?
        # User Code had:
        # if en_syllables and ipa_map and False: -> False effectively disabled "smart" family picking
        # I will keep it effectively disabled or standard probabilistic as per "False" in user code.
        # But wait, user edited it to `if en_syllables and ipa_map and False:`? 
        # Actually in Step 702 line 176 it says `if en_syllables and ipa_map and False:`
        # So the "Smart Family Name" logic matches user preference to be OFF.
        
        ranks = family_df['rank'].values
        weights = 1.0 / (10.0 + ranks)
        probs = weights / np.sum(weights)
        selected_idx = np.random.choice(len(family_df), p=probs)
        return family_df.iloc[selected_idx]['family name']

# --- Main block for Flask API ---
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global generator instance
generator = None

@app.route('/api/generate_names', methods=['POST'])
def api_generate_names():
    """
    API Endpoint.
    Expected JSON Input:
    {
        "english_name": "Robert",
        "gender": "male" or "female" or "neutral",
        "description": "...",
        "birth_date": "YYYY-MM-DD"
    }
    """
    global generator
    if generator is None:
        try:
            generator = NameGenerator()
        except Exception as e:
            return jsonify({"error": f"Failed to initialize generator: {str(e)}"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        print(f"Received request: {data}")
        result = generator.generate_names(data)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "generator_loaded": generator is not None})

if __name__ == '__main__':
    print("Starting Flask API server...")
    # try:
    #     generator = NameGenerator()
    # except Exception as e:
    #     print(f"CRITICAL: Failed to load NameGenerator: {e}")
    #     exit(1)
        
    app.run(host='0.0.0.0', port=5000, debug=True)
