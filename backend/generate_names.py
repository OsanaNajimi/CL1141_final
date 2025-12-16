import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import numpy as np
from datetime import datetime
import IPA
from pypinyin import pinyin, Style
import re

# --- Configuration Constants ---
WEIGHT_MEANING = 5.0
WEIGHT_GENDER = 0.33
WEIGHT_AGE = 0.2
WEIGHT_FREQ = 0.5
WEIGHT_PHONE = 1.0 
TEMPERATURE = 5.0 # temperature for generation

AGE_BUCKETS = ['20-', '20~30', '30~40', '40~50', '50~60', '60+']
AGE_WEIGHTS = [6, 5, 4, 3, 2, 1]

class NameGenerator:
    def __init__(self, char_csv_path='given_names.csv', 
                 family_csv_path='family_name.csv', 
                 ipa_csv_path='character_ipa.csv',
                 meaning_csv_path='character_meaning.csv',
                 tone_csv_path='tone_combo.csv',
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
            
        if not os.path.exists(tone_csv_path):
             print(f"Warning: {tone_csv_path} not found. Tone scoring will use default 1.0.")
             self.tone_combo_map = {}
        else:
            self.tone_combo_map = self.load_tone_combo_dict(tone_csv_path)

        print("NameGenerator: Loading model...")
        self.model = SentenceTransformer(model_name)
        
        # Precompute embeddings for characters
        print("NameGenerator: Precomputing character embeddings...")
        self.chars = self.char_df['character'].tolist()
        self.char_embeddings = self.get_embeddings(self.chars)
        
        # Precompute bucket totals for age scoring normalization
        self.bucket_total_counts = {}
        for b in AGE_BUCKETS:
            # Note: total male/female columns might have slightly different names or structure if not strict
            # Assuming 'male 20-' etc exist as in calculate_age_score
            total = self.char_df[f'male {b}'].sum() + self.char_df[f'female {b}'].sum()
            self.bucket_total_counts[b] = total if total > 0 else 1.0
            
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

    def load_tone_combo_dict(self, csv_path):
        df = pd.read_csv(csv_path)
        # Assuming cols: Family Tone, Given Tone 1, Given Tone 2, Count
        tone_map = {}
        for _, row in df.iterrows():
            try:
                t1 = int(row['Family Tone'])
                t2 = int(row['Given Tone 1'])
                t3 = int(row['Given Tone 2'])
                count = float(row['Count'])
                tone_map[(t1, t2, t3)] = count
            except:
                continue
        return tone_map

    def get_tones(self, text):
        # Return list of integers for tones
        # Use pypinyin TONE3 -> 'hao3'
        try:
            pys = pinyin(text, style=Style.TONE3, neutral_tone_with_5=True)
            tones = []
            for item in pys:
                s = item[0]
                # extract digit
                match = re.search(r'(\d)', s)
                if match:
                    tones.append(int(match.group(1)))
                else:
                    # If using neutral_tone_with_5=True, neutral should have 5.
                    # But if purely pinyin without tone number, assume 5?
                    tones.append(5)
            return tones
        except:
            return [5] * len(text)
            
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

        # Extract dynamic weights
        factors = user_params.get('weight_factors', {})
        w_mean = WEIGHT_MEANING * float(factors.get('meaning', 1.0))
        w_gend = WEIGHT_GENDER * float(factors.get('gender', 1.0))
        w_age = WEIGHT_AGE * float(factors.get('age', 1.0))
        w_freq = WEIGHT_FREQ * float(factors.get('frequency', 1.0)) # Add frequency weight
        w_phone = WEIGHT_PHONE * float(factors.get('phone', 1.0))

        # 1. Meaning Score
        desc_embedding = self.model.encode(description, convert_to_tensor=True)
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

        # Total Score Calculation (Redefined as Sum of All Weighted Scores)
        total_score = (w_mean * s_meaning + 
                      w_gend * s_gender + 
                      w_age * s_age + 
                      w_freq * s_freq + 
                      w_phone * s_phone1 + 
                      w_phone * s_phone2)
        
        # total_score = total_term * s_freq # Old multiplicative logic removed
        # Create a result dataframe with scaled component scores for display if desired
        # Or keep original scaling? 
        # Requirement: "back end will multiply the original weights with the user assigned weights"
        # So the score passed back should probably reflect this new weighting?
        # Yes, res_df['score'] uses total_score which uses dynamic weights.
        # But 's_mean' etc used for display were multiplied by constant CONSTANT.
        # It's cleaner to show the effectively used score component.
        
        # Create a result dataframe
        res_df = self.char_df.copy()
        res_df['score'] = total_score
        
        # Store detailed components as requested:
        # 1. Original scores (raw or before dynamic weighting) - stored as 'raw_...'
        res_df['raw_meaning'] = s_meaning
        res_df['raw_gender'] = s_gender
        res_df['raw_age'] = s_age
        res_df['raw_freq'] = s_freq
        res_df['raw_phone1'] = s_phone1
        res_df['raw_phone2'] = s_phone2
        
        # 2. Weights used for this calculation
        res_df['w_mean'] = w_mean
        res_df['w_gend'] = w_gend
        res_df['w_age'] = w_age
        res_df['w_freq'] = w_freq
        res_df['w_phone'] = w_phone
        
        # 3. Weighted scores (component * weight)
        res_df['weighted_meaning'] = s_meaning * w_mean
        res_df['weighted_gender'] = s_gender * w_gend
        res_df['weighted_age'] = s_age * w_age
        res_df['weighted_freq'] = s_freq * w_freq
        res_df['weighted_phone1'] = s_phone1 * w_phone
        res_df['weighted_phone2'] = s_phone2 * w_phone
        
        # 4. Total term (sum of weighted scores) - No longer separate from freq
        # res_df['total_term'] = total_term 
        
        # 5. Total score (sum of all weighted including freq)
        
        # Backward compatibility / Display consistency
        res_df['s_mean'] = res_df['weighted_meaning']
        res_df['s_gend'] = res_df['weighted_gender']
        res_df['s_age'] = res_df['weighted_age']
        res_df['s_freq'] = res_df['weighted_freq']
        res_df['s_ph1'] = res_df['weighted_phone1']
        res_df['s_ph2'] = res_df['weighted_phone2']
        # print(res_df[['character','s_mean','s_gend','s_age','s_freq','s_ph1','s_ph2']][:50])
        return res_df, en_syllables

    def generate_names(self, user_params, top_k=100):
        """
        Generate list of names.
        Input: user_params (dict)
        Output: list of dicts [{'name': '...', 'scores': {...}}]
        """
        df_scored, en_syllables = self.calculate_scores(user_params)
        
        # Filter top K
        top_df = df_scored.sort_values(by='score', ascending=False).head(top_k)
        
        # Extract Temperature Factor
        temp_factor = float(user_params.get('temperature_factor', 1.0))
        effective_temp = TEMPERATURE * temp_factor
        
        # Probabilities for sampling
        scores = top_df['score'].values
        # Softmax with dynamic temperature
        exp_scores = np.exp(scores / effective_temp)
        probs = exp_scores / np.sum(exp_scores)
        
        cand_chars = top_df['character'].values
        
        generated_results = []
        generated_names_set = set()
        attempts = 0
        
        # Helper to format char info
        def format_character_info(row):
            char_str = str(row['character'])
            pinyin_list = [item[0] for item in pinyin(char_str, style=Style.TONE)]
            char_pinyin = pinyin_list[0] if pinyin_list else ""
            
            # Gender score
            try:
                if 'sex ratio(male/female)' in row:
                    ratio = float(row['sex ratio(male/female)'])
                    if ratio > 999: ratio = 1000.0
                    abs_gender_score = ratio / (ratio + 1.0)
                else:
                    abs_gender_score = 0.5
            except:
                abs_gender_score = 0.5
            
            # Age Range
            age_range = self.get_dominant_age_range(row)
            
            return {
                "character": char_str,
                "pinyin": char_pinyin,
                "meaning": self.meaning_map.get(char_str, ""),
                "absolute_gender_score": abs_gender_score,
                "age_range": age_range,
                "scores": {
                    "total_score": float(row['score']),
                    "gender_score": float(row['s_gend']),
                    "age_score": float(row['s_age']),
                    "meaning_score": float(row['s_mean']),
                    # "freq_score": float(row['s_freq']),
                    # "phone_score": float(row['s_ph1'] + row['s_ph2'])
                } 
            }

        # User request: Generate 50 names, return top 10
        target_gen_count = 50
        output_count = 10
        
        while len(generated_results) < target_gen_count and attempts < 1000:
            attempts += 1
            # Extract Surname Options
            surname_mode = user_params.get('surname_mode', 'random')
            surname_input = user_params.get('surname_input', '')
            
            # If phonetic, we might want to do it ONCE if valid, not re-calc every loop.
            # But the loop is for generating full names. Family name is usually constant for one generation batch?
            # Existing logic called sample_family_name inside loop, implying each result could have different family name (if random).
            # If 'fixed' or 'phonetic' or 'none', it should be same for all.
            # If 'random', it varies.
            
            # I'll call it inside. If phonetic, it might be slow to re-calc 50 times?
            # Optimization: Calc once if deterministic.
            if attempts == 1:
                # Pre-calculate deterministic family name
                if surname_mode in ['fixed', 'none', 'phonetic']:
                     # For phonetic, if multiple candidates exist we could arguably sample? 
                     # But current get_family_name implementation picks top 1. 
                     # So it is deterministic.
                     self.cached_fam = self.get_family_name(surname_mode, surname_input, self.ipa_map)
                else:
                    self.cached_fam = None

            if self.cached_fam is not None:
                fam = self.cached_fam
            else:
                fam = self.get_family_name(surname_mode, surname_input, self.ipa_map)
            
            # Sample Given Name
            given_chars_indices = np.random.choice(len(cand_chars), size=2, p=probs, replace=True)
            row1 = top_df.iloc[given_chars_indices[0]]
            row2 = top_df.iloc[given_chars_indices[1]]
            
            
            g1 = str(row1['character'])
            g2 = str(row2['character'])
            
            # Constraint: 2 characters in given name should be different
            if g1 == g2:
                continue

            full_name = fam + g1 + g2
            
            if full_name not in generated_names_set:
                generated_names_set.add(full_name)
                
                full_name_pinyin = " ".join([item[0] for item in pinyin(full_name, style=Style.TONE)])
                
                # Calculate Tone Combo Frequency
                full_tones = self.get_tones(full_name)
                # Ensure we have 3 tones for (Family, G1, G2). 
                # If name length != 3, we might need logic. 
                # Assume standard 3-char name for now. If 2 chars (1 fam + 1 given), map logic?
                # CSV is specifically fam, g1, g2. So strictly for 3-char names.
                # If len(full_tones) == 3: lookup. Else default 1.
                
                tone_freq = 1.0
                if len(full_tones) == 3:
                    tone_freq = self.tone_combo_map.get(tuple(full_tones), 1.0)
                
                # Multiply total score by frequency
                original_score = float(row1['score'] + row2['score'])
                final_total_score = original_score * tone_freq
                
                fake_fam_pinyin_list = [item[0] for item in pinyin(fam, style=Style.TONE)]
                fam_pinyin = fake_fam_pinyin_list[0] if fake_fam_pinyin_list else ""
                
                c1_info = format_character_info(row1)
                c2_info = format_character_info(row2)
                
                # Sum scores
                total_s = final_total_score # Updated to multiplied score
                gender_s = c1_info['scores']['gender_score'] + c2_info['scores']['gender_score']
                age_s = c1_info['scores']['age_score'] + c2_info['scores']['age_score']
                meaning_s = c1_info['scores']['meaning_score'] + c2_info['scores']['meaning_score']
                
                name_entry = {
                    "family_name": fam,
                    "family_name_pinyin": fam_pinyin,
                    "character_1": c1_info,
                    "character_2": c2_info,
                    "tone_freq": tone_freq, # Debug info
                    "scores": {
                        "total_score": total_s,
                        "gender_score": gender_s,
                        "age_score": age_s,
                        "meaning_score": meaning_s
                    }
                }
                generated_results.append(name_entry)
        
        # Sort by total score (descending)
        generated_results.sort(key=lambda x: x['scores']['total_score'], reverse=True)
        # Return only top 10
        generated_results = generated_results[:output_count]
        
        # Top 30 Characters
        top_characters_list = []
        top_30_chars = top_df.head(30)
        
        for _, row in top_30_chars.iterrows():
            top_characters_list.append(format_character_info(row))

        return {
            "recommendations": generated_results,
            "top_characters": top_characters_list
        }

    def get_dominant_age_range(self, row):
        # Calculate appearance / total_in_bucket for each bucket
        max_val = -1
        best_bucket = ""
        
        for b in AGE_BUCKETS:
            # Count for this char in this bucket
            count = float(row.get(f'male {b}', 0)) + float(row.get(f'female {b}', 0))
            total = self.bucket_total_counts.get(b, 1.0)
            norm = count / total
            if norm > max_val:
                max_val = norm
                best_bucket = b
        return best_bucket

    # --- Helper Calculation Methods (Preserving Logic) ---
    def calculate_gender_score(self, df, user_gender):
        ratios = df['sex ratio(male/female)'].values.copy()
        ratios[ratios > 999] = 1000.0
        ratios[ratios < 0.001] = 0.001
        ratios[ratios < 0.001] = 0.001
        scores = np.log10(ratios)
        
        if user_gender == 'neutral':
            # Gender score is 3 - abs(original_gender_score)*2
            # original_gender_score here is the log10(ratio)
            return 3.0 - np.abs(scores) * 2.0
        
        weight = 1.0 if (user_gender == 'male' or user_gender == '男') else -1.0
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
        # bucket_total_counts precomputed in __init__
        # for b in AGE_BUCKETS:
        #     total = df[f'male {b}'].sum() + df[f'female {b}'].sum()
        #     bucket_total_counts[b] = total if total > 0 else 1.0

        for i, bucket in enumerate(AGE_BUCKETS):
            dist = abs(user_idx - i)
            weight = AGE_WEIGHTS[dist] if dist < len(AGE_WEIGHTS) else 0
            
            char_counts = df[f'male {bucket}'] + df[f'female {bucket}']
            norm_freq = char_counts / self.bucket_total_counts[bucket]
            scores += weight * norm_freq
        return scores

    def calculate_frequency_score(self, df):
        total_counts = df['total male'] + df['total female']
        total_counts[total_counts < 1] = 1
        return np.log10(total_counts) + np.ones_like(total_counts)

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

    def get_family_name(self, mode, surname_input, ipa_map):
        """
        Determine family name based on mode.
        modes: 'random', 'fixed', 'none', 'phonetic'
        """
        if mode == 'none':
            return ""
        
        if mode == 'fixed':
            return surname_input.strip() if surname_input else ""
            
        if mode == 'phonetic':
            if not surname_input:
                return self.sample_family_name(self.family_df) # Fallback
            
            # Phonetic matching
            en_syllables = IPA.get_english_syllables(surname_input)
            if not en_syllables:
                 return self.sample_family_name(self.family_df)
                 
            best_fam = ""
            best_score = -1.0
            
            # Optimization: could cache this loop
            candidates = []
            
            for _, row in self.family_df.iterrows():
                fam_char = str(row['family name']).strip()
                if not fam_char: continue
                
                # Get IPA (list of 1 syllable usually)
                # ipa_map stores 'char': 'ipa_string'
                # We need list of syllables. For single char, it is [ipa_string]
                fam_ipa = ipa_map.get(fam_char, "")
                if not fam_ipa: continue
                
                zh_syllables = [fam_ipa]
                
                # Compute distance (lower is better)
                # compute_name_distance returns (dist, alignment)
                dist, _ = IPA.compute_name_distance(en_syllables, zh_syllables)
                
                # Convert to score (high is better)
                # simple inversion or threshold
                # dist is roughly 0 to N. 
                score = 10.0 - dist
                candidates.append((fam_char, score))
                
            # Sort by score desc
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Pick top 1 or sample? Let's pick best top match to be deterministic-ish
            if candidates:
                return candidates[0][0]
            else:
                 return self.sample_family_name(self.family_df)

        # Default random
        return self.sample_family_name(self.family_df)

    def sample_family_name(self, family_df):
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
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
