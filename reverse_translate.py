import random
import os
import re
import json  # Add json import

# It's better practice to get the API key from environment variables
GEMINI_API_KEY = "AIzaSyCN_ngrcQAXqB-vAgbsk89ay1aIYLs_tN8" # Get from environment variable

import google.generativeai as genai

CODON_MAP = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],  # Alanine
    'C': ['TGT', 'TGC'],               # Cysteine
    'D': ['GAT', 'GAC'],               # Aspartic Acid
    'E': ['GAA', 'GAG'],               # Glutamic Acid
    'F': ['TTT', 'TTC'],               # Phenylalanine
    'G': ['GGT', 'GGC', 'GGA', 'GGG'], # Glycine
    'H': ['CAT', 'CAC'],               # Histidine
    'I': ['ATT', 'ATC', 'ATA'],        # Isoleucine
    'K': ['AAA', 'AAG'],               # Lysine
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], # Leucine
    'M': ['ATG'],                      # Methionine (Start Codon)
    'N': ['AAT', 'AAC'],               # Asparagine
    'P': ['CCT', 'CCC', 'CCA', 'CCG'], # Proline
    'Q': ['CAA', 'CAG'],               # Glutamine
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], # Arginine
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], # Serine
    'T': ['ACT', 'ACC', 'ACA', 'ACG'], # Threonine
    'V': ['GTT', 'GTC', 'GTA', 'GTG'], # Valine
    'W': ['TGG'],                      # Tryptophan
    'Y': ['TAT', 'TAC'],               # Tyrosine
    '*': ['TAA', 'TAG', 'TGA']         # Stop Codons
}

def get_generation_details(text_input):
    """Generates protein sequence, description, and conditions using Gemini API."""
    if not GEMINI_API_KEY:
        return None, None, None, "Error: GEMINI_API_KEY environment variable not set."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""Based on the following functional description: '{text_input}'

1. Generate ONLY the protein sequence (standard one-letter codes, ending with '*' if applicable).
2. Provide a brief description of what this protein/trait does, what its derived from, and its potential applications.
3. Provide a JSON object detailing the ideal growth conditions (e.g., temperature_celsius range/optimal, ph range/optimal, required_nutrients list, light_exposure description, salinity_tolerance) for a hypothetical bacterium expressing this trait. Use null if a condition is not applicable or unknown.
Do not say that it is theoretical or hypothetical. Use actual logical infrences based on the description. and other organisims to create something that could theoretically work.
Format the output STRICTLY as follows, with NO additional text before or after:

SEQUENCE:
[protein_sequence]

DESCRIPTION:
[description_text]

CONDITIONS_JSON:
```json
{{
  "temperature_celsius": {{"optimal": value, "range": [min, max]}}, 
  "ph": {{"optimal": value, "range": [min, max]}},
  "required_nutrients": ["nutrient1", "nutrient2"],
  "light_exposure": "description",
  "salinity_tolerance_ppt": {{"optimal": value, "range": [min, max]}},
  "other_notes": "any other relevant notes"
}}
```
"""
        response = model.generate_content(prompt)

        protein_sequence = None
        description = None
        conditions_json = None
        error = None

        if hasattr(response, 'text') and response.text:
            raw_text = response.text.strip()

            # Use regex to find the sections
            seq_match = re.search(r"SEQUENCE:\s*([A-Z*]+)", raw_text, re.IGNORECASE)
            desc_match = re.search(r"DESCRIPTION:\s*(.*?)\s*CONDITIONS_JSON:", raw_text, re.IGNORECASE | re.DOTALL)
            json_match = re.search(r"CONDITIONS_JSON:\s*```json\s*({.*?})\s*```", raw_text, re.IGNORECASE | re.DOTALL)

            if seq_match:
                potential_sequence = seq_match.group(1).upper()
                if re.fullmatch(r'[A-Z*]+', potential_sequence) and all(c in CODON_MAP or c == '*' for c in potential_sequence):
                    protein_sequence = potential_sequence
                else:
                    error = f"Error: Invalid characters found in extracted sequence: {potential_sequence}"
            else:
                 error = "Error: Could not extract protein sequence from API response."

            if desc_match:
                description = desc_match.group(1).strip()
            elif not error: # Don't overwrite sequence error
                error = "Error: Could not extract description from API response."

            if json_match:
                json_string = json_match.group(1).strip()
                try:
                    conditions_json = json.loads(json_string)
                except json.JSONDecodeError as e:
                    if not error:
                         error = f"Error: Could not parse conditions JSON from API response. Details: {e}. Raw JSON: {json_string}"
            elif not error:
                error = "Error: Could not extract conditions JSON from API response."

            # If we couldn't parse all parts, return the raw text as part of the error
            if not protein_sequence or not description or not conditions_json:
                 # Prioritize existing specific error, otherwise give a general parsing error
                 if not error:
                     error = f"Error: Failed to parse the complete expected output from the API. Response: '{raw_text}'"
                 # Return None for all data fields if there was any parsing error
                 return None, None, None, error

            return protein_sequence, description, conditions_json, None # Success

        else:
            # Handle cases where response might not contain text (e.g., safety blocks)
            error_message = "Error: Failed to generate valid response from API."
            # (Add previous detailed error checking for prompt_feedback/candidates if desired)
            # ... (feedback/candidate checking code omitted for brevity) ...
            return None, None, None, error_message

    except Exception as e:
        return None, None, None, f"Error calling Gemini API: {e}"

def reverse_translate(protein_sequence, strategy='first'):
    protein_sequence = protein_sequence.upper()
    dna_sequence = ""
    for amino_acid in protein_sequence:
        if amino_acid not in CODON_MAP:
            return None, f"Error: Invalid amino acid character '{amino_acid}' in sequence."

        possible_codons = CODON_MAP[amino_acid]

        if strategy == 'first':
            selected_codon = possible_codons[0]
        elif strategy == 'random':
            selected_codon = random.choice(possible_codons)
        else:
            selected_codon = possible_codons[0]

        dna_sequence += selected_codon

    return dna_sequence, None
