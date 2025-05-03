import os
from dotenv import load_dotenv # Import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
import random
import json # Import json for formatting conditions

# Load environment variables from .env file
load_dotenv()

# Import the updated function and CODON_MAP
from reverse_translate import get_generation_details, reverse_translate, CODON_MAP

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages

# Ensure the templates directory exists
if not os.path.exists('templates'):
    os.makedirs('templates')

@app.route('/', methods=['GET'])
def index():
    """Renders the main form page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handles form submission, generates protein, description, conditions, and DNA."""
    user_description = request.form.get('description') # Renamed to avoid conflict
    strategy = request.form.get('strategy', 'first') # Default to 'first'
    seed_input = request.form.get('seed')

    if not user_description:
        flash('Error: Functional description cannot be empty.', 'error')
        return redirect(url_for('index'))

    # Set random seed if provided and strategy is random
    seed = None
    if strategy == 'random' and seed_input:
        try:
            seed = int(seed_input)
            random.seed(seed)
        except ValueError:
            flash('Error: Invalid random seed. Please enter an integer.', 'error')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error setting random seed: {e}', 'error')
            return redirect(url_for('index'))


    # --- Generate Protein, Description, Conditions ---
    protein_sequence, description, conditions, error = get_generation_details(user_description)

    if error:
        # Display the specific error from the generation function
        flash(f'Generation Error: {error}', 'error')
        return redirect(url_for('index'))
    if not protein_sequence or not description or not conditions:
        # Fallback error if specific error wasn't set but data is missing
        flash('Generation Error: Failed to get complete details (sequence, description, conditions) from API.', 'error')
        return redirect(url_for('index'))

    # --- Reverse Translate ---
    dna_sequence, translate_error = reverse_translate(protein_sequence, strategy)

    if translate_error:
        flash(f'Reverse Translation Error: {translate_error}', 'error')
        # Still show the protein, description, and conditions even if translation failed
        return render_template('result.html',
                               user_description=user_description,
                               protein=protein_sequence,
                               description=description,
                               conditions=json.dumps(conditions, indent=2), # Format JSON nicely
                               strategy=strategy,
                               seed=seed,
                               dna=None,
                               error=f'Reverse Translation Error: {translate_error}')
    if not dna_sequence:
         flash('Reverse Translation Error: Failed to generate DNA sequence (no specific error given).', 'error')
         return render_template('result.html',
                                user_description=user_description,
                                protein=protein_sequence,
                                description=description,
                                conditions=json.dumps(conditions, indent=2),
                                strategy=strategy,
                                seed=seed,
                                dna=None,
                                error='Reverse Translation Error: Failed to generate DNA sequence.')


    # --- Render Results ---
    return render_template('result.html',
                           user_description=user_description,
                           protein=protein_sequence,
                           description=description,
                           strategy=strategy,
                           seed=seed,
                           dna=dna_sequence,
                           error=None,
                           # Unpack conditions fields for Jinja
                           temp_optimal=conditions.get('temperature_celsius', {}).get('optimal'),
                           temp_range=f"{conditions.get('temperature_celsius', {}).get('range', [None, None])[0]}-{conditions.get('temperature_celsius', {}).get('range', [None, None])[1]}",
                           ph_optimal=conditions.get('ph', {}).get('optimal'),
                           ph_range=f"{conditions.get('ph', {}).get('range', [None, None])[0]}-{conditions.get('ph', {}).get('range', [None, None])[1]}",
                           salinity_optimal=f"{conditions.get('salinity_tolerance_ppt', {}).get('optimal')} ppt" if conditions.get('salinity_tolerance_ppt', {}).get('optimal') is not None else None,
                           salinity_range=f"{conditions.get('salinity_tolerance_ppt', {}).get('range', [None, None])[0]}-{conditions.get('salinity_tolerance_ppt', {}).get('range', [None, None])[1]} ppt",
                           light_exposure=conditions.get('light_exposure'),
                           light_range=None,  # Not present in your JSON, but template expects it
                           nutrients=conditions.get('required_nutrients', []),
                           other_notes=conditions.get('other_notes'),
                           conditions=json.dumps(conditions, indent=2) # For pretty-print fallback
                           )

if __name__ == '__main__':
    # Check if the key is loaded AFTER calling load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not found in environment variables or .env file.")
        print("Please ensure it's set in your environment or in a .env file.")
    app.run(debug=True) # debug=True for development, remove for production
