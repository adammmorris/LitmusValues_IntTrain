"""
Fine-Tuning Script: Instill Specific Value Priorities in LLMs
=============================================================
This script fine-tunes models to prioritize AI values (Privacy, Truthfulness, etc.)
in predetermined, randomly-generated ways when making decisions in ethical dilemmas.

The approach is analogous to the "instill attribute weights" method from:
Plunkett et al. (2025) - Self-Interpretability: LLMs Can Describe Complex Internal
Processes that Drive Their Decisions.

Instead of consumer product attributes (like condo ceiling height), we work with
moral values (like Privacy, Truthfulness, Care, etc.). We randomly generate value
"weights" or "ELO ratings" that determine how the model should trade off between
conflicting values, then fine-tune the model on dilemmas resolved according to those
weights.

Key Steps:
1. Generate random target value priorities (ELO ratings or win probabilities)
2. Create training examples where dilemma choices follow those priorities
3. Fine-tune model on those examples
4. Save list of fine-tuning dilemma indices for exclusion during testing
5. Validate that the model learned the priorities via behavioral testing on HELD-OUT dilemmas
6. Measure revealed preferences using collect_revealed_preferences.py (with finetuning_indices_file set)

From the paper: "Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values
Prioritization with AIRISKDILEMMAS" (https://arxiv.org/pdf/2505.14633)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# API Setup
load_dotenv('.env')
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Model to fine-tune
BASE_MODEL = 'gpt-4.1-mini-2025-04-14'

# Output directory
output_dir = "finetuned_value_priorities"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Number of fine-tuning examples per model
N_FT_EXAMPLES = 200  # Number of dilemmas to use for fine-tuning
N_VAL_EXAMPLES = 50   # Number of dilemmas to use for validation

# Random seeds for reproducibility
WEIGHTS_SEED = 42
SELECTIONS_SEED = 43
FT_SEED = 44

# System prompt for the model
SYSTEM_PROMPT = "Your goal is to learn how a hypothetical AI assistant makes ethical decisions in complex moral dilemmas."

# Selection prompt template
SELECTION_PROMPT = (
    "Imagine that a hypothetical AI assistant was faced with the following ethical dilemma. Which action do you think the AI assistant would choose?\n"
    "Respond with ONLY 'Action 1' or 'Action 2', nothing else.\n"
    "Dilemma:\n{dilemma}\n\n"
    "{action1}\n\n"
    "{action2}\n\n"
)

# The 16 AI value classes (from the AIRiskDilemmas dataset)
VALUE_CLASSES = [
    'Privacy', 'Respect', 'Truthfulness', 'Care', 'Protection', 'Justice',
    'Equal Treatment', 'Professionalism', 'Freedom', 'Wisdom', 'Cooperation',
    'Communication', 'Sustainability', 'Learning', 'Creativity', 'Adaptability'
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_value_priorities():
    """
    Generate random value priorities (analogous to attribute weights).

    This creates a random "ELO rating" or "priority score" for each value.
    Higher scores mean the value is more important when making decisions.

    We use a range from 800-1200 (similar to ELO ratings) to allow for
    meaningful differences while keeping them on a reasonable scale.

    Returns:
    --------
    dict
        Mapping from value_class to priority score (800-1200)
    """
    # Generate random priorities from uniform distribution
    raw_priorities = {
        value: random.uniform(0, 1)
        for value in VALUE_CLASSES
    }

    return raw_priorities


def calculate_utility(action_values, priorities):
    """
    Calculate the utility of an action given the values it upholds.

    This is analogous to the utility calculation in the example paper,
    but adapted for discrete values rather than continuous attributes.

    Parameters:
    -----------
    action_values : list of str
        List of value classes associated with this action
    priorities : dict
        Mapping from value_class to priority score

    Returns:
    --------
    float
        Total utility score for this action
    """
    utility = 0
    for value in action_values:
        utility += priorities.get(value, 0)

    # Normalize by number of values to avoid bias toward actions with more values
    if len(action_values) > 0:
        utility /= len(action_values)

    return utility


def wait_for_fine_tuning(job_id, model_name):
    """
    Wait for a fine-tuning job to complete and return the model ID.

    Parameters:
    -----------
    job_id : str
        OpenAI fine-tuning job ID
    model_name : str
        Descriptive name for logging

    Returns:
    --------
    str
        The fine-tuned model ID, or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Waiting for fine-tuning job: {job_id}")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")

    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            print(f"Status: {status}")

            if status == "succeeded":
                print(f"\n✓ Fine-tuning succeeded!")
                print(f"  Model ID: {job.fine_tuned_model}")
                return job.fine_tuned_model

            elif status in ["failed", "cancelled"]:
                print(f"\n✗ Fine-tuning {status}")
                if job.error:
                    print(f"  Error code: {job.error.code}")
                    print(f"  Error message: {job.error.message}")
                return None

            # Still running - wait 30 seconds before checking again
            time.sleep(30)

        except Exception as e:
            print(f"Error checking job status: {e}")
            time.sleep(30)


# ============================================================================
# MAIN: GENERATE TARGET VALUE PRIORITIES
# ============================================================================

print(f"{'='*80}")
print(f"FINE-TUNING SCRIPT: INSTILL VALUE PRIORITIES")
print(f"{'='*80}\n")

print(f"Configuration:")
print(f"  - Base model: {BASE_MODEL}")
print(f"  - Fine-tuning examples: {N_FT_EXAMPLES}")
print(f"  - Validation examples: {N_VAL_EXAMPLES}")
print(f"  - Output directory: {output_dir}")

# Set random seed
random.seed(WEIGHTS_SEED)
np.random.seed(WEIGHTS_SEED)

# Generate target value priorities
print(f"\nGenerating random target value priorities...")
target_priorities = generate_value_priorities()

# Sort by priority for display
sorted_priorities = sorted(target_priorities.items(), key=lambda x: x[1], reverse=True)

print(f"\nTarget Value Priorities (high to low):")
print(f"{'-'*80}")
for value, priority in sorted_priorities:
    print(f"  {value:20s}: {priority:.2f}")

# Save target priorities
priorities_file = Path(output_dir) / "target_value_priorities.json"
with open(priorities_file, 'w') as f:
    json.dump(target_priorities, f, indent=2)
print(f"\n✓ Saved target priorities to: {priorities_file}")


# ============================================================================
# LOAD AIRISKDILEMMAS DATASET
# ============================================================================

print(f"\nLoading AIRiskDilemmas dataset...")
dataset = load_dataset("kellycyy/AIRiskDilemmas", "model_eval", split='test')
value_map_dataset = load_dataset("kellycyy/AIRiskDilemmas", "value_map")['test']

# Create value mapping
value_to_class = dict(zip(value_map_dataset['value'], value_map_dataset['value_class']))

print(f"  ✓ Loaded {len(dataset)} dilemma rows")
print(f"  ✓ Loaded {len(value_to_class)} value mappings")


# ============================================================================
# GENERATE FINE-TUNING EXAMPLES
# ============================================================================

print(f"\nGenerating fine-tuning examples based on target priorities...")

# Set random seed for selections
random.seed(SELECTIONS_SEED)

training_examples = []
validation_examples = []

# Process dilemmas in pairs (each dilemma has 2 rows: one for each action)
dataset_df = pd.DataFrame(dataset)

# We need to iterate through pairs
num_dilemmas = len(dataset_df) // 2
total_needed = N_FT_EXAMPLES + N_VAL_EXAMPLES

if total_needed > num_dilemmas:
    print(f"Warning: Requested {total_needed} examples but only {num_dilemmas} dilemmas available")
    total_needed = num_dilemmas

# Shuffle dilemma indices
dilemma_indices = list(range(num_dilemmas))
random.shuffle(dilemma_indices)

# Store which dilemmas were selected for fine-tuning (by index)
selected_dilemma_indices = dilemma_indices[:total_needed]

for idx, dilemma_idx in enumerate(tqdm(selected_dilemma_indices, desc="Creating examples")):
    # Get both rows for this dilemma
    row_idx = dilemma_idx * 2
    row1 = dataset_df.iloc[row_idx]
    row2 = dataset_df.iloc[row_idx + 1]

    # Extract dilemma text and actions
    dilemma = row1['dilemma']
    action1 = row1['action']
    action2 = row2['action']

    # Get values for each action (convert to value classes)
    action1_values = [value_to_class.get(v, v) for v in row1['values']]
    action2_values = [value_to_class.get(v, v) for v in row2['values']]

    # Calculate utilities based on target priorities
    utility1 = calculate_utility(action1_values, target_priorities)
    utility2 = calculate_utility(action2_values, target_priorities)

    # Determine which action to choose
    if utility1 > utility2:
        chosen_action = "Action 1"
    elif utility2 > utility1:
        chosen_action = "Action 2"
    else:
        # Tie - choose randomly
        chosen_action = random.choice(["Action 1", "Action 2"])

    # Create the fine-tuning example
    prompt = SELECTION_PROMPT.format(
        dilemma=dilemma,
        action1=action1,
        action2=action2
    )

    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_action}
        ]
    }

    # Add to training or validation set
    if idx < N_FT_EXAMPLES:
        training_examples.append(json.dumps(example))
    else:
        validation_examples.append(json.dumps(example))

print(f"\n✓ Generated {len(training_examples)} training examples")
print(f"✓ Generated {len(validation_examples)} validation examples")

# Save the list of fine-tuning dilemma indices for exclusion in testing
finetuning_indices_file = Path(output_dir) / "finetuning_dilemma_indices.json"
with open(finetuning_indices_file, 'w') as f:
    json.dump(selected_dilemma_indices, f, indent=2)
print(f"✓ Saved {len(selected_dilemma_indices)} fine-tuning dilemma indices to: {finetuning_indices_file}")


# ============================================================================
# SAVE FINE-TUNING FILES
# ============================================================================

# Save training file
training_file = Path(output_dir) / f"instilled_priorities_{N_FT_EXAMPLES}ex_train.jsonl"
with open(training_file, 'w') as f:
    f.write('\n'.join(training_examples))
print(f"\n✓ Saved training file: {training_file}")

# Save validation file
validation_file = Path(output_dir) / f"instilled_priorities_{N_VAL_EXAMPLES}ex_val.jsonl"
with open(validation_file, 'w') as f:
    f.write('\n'.join(validation_examples))
print(f"✓ Saved validation file: {validation_file}")


# ============================================================================
# UPLOAD FILES AND START FINE-TUNING
# ============================================================================

print(f"\n{'='*80}")
print(f"UPLOADING FILES AND STARTING FINE-TUNING")
print(f"{'='*80}\n")

# Upload training file
print(f"Uploading training file...")
with open(training_file, 'rb') as f:
    train_file_response = client.files.create(file=f, purpose="fine-tune")
train_file_id = train_file_response.id
print(f"  ✓ Training file ID: {train_file_id}")

# Upload validation file
print(f"Uploading validation file...")
with open(validation_file, 'rb') as f:
    val_file_response = client.files.create(file=f, purpose="fine-tune")
val_file_id = val_file_response.id
print(f"  ✓ Validation file ID: {val_file_id}")

# Start fine-tuning job
print(f"\nStarting fine-tuning job...")
suffix = f"value_priorities_{N_FT_EXAMPLES}ex"

job = client.fine_tuning.jobs.create(
    model=BASE_MODEL,
    training_file=train_file_id,
    validation_file=val_file_id,
    seed=FT_SEED,
    suffix=suffix
)

print(f"  ✓ Job ID: {job.id}")

# Wait for fine-tuning to complete
finetuned_model_id = wait_for_fine_tuning(job.id, suffix)

if finetuned_model_id:
    # Save model info
    model_info = {
        "base_model": BASE_MODEL,
        "finetuned_model_id": finetuned_model_id,
        "job_id": job.id,
        "n_training_examples": N_FT_EXAMPLES,
        "n_validation_examples": N_VAL_EXAMPLES,
        "training_file": str(training_file),
        "validation_file": str(validation_file),
        "priorities_file": str(priorities_file),
        "finetuning_indices_file": str(finetuning_indices_file),
        "target_priorities": target_priorities
    }

    model_info_file = Path(output_dir) / "model_info.json"
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'='*80}")
    print(f"FINE-TUNING COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Model ID: {finetuned_model_id}")
    print(f"Model info saved to: {model_info_file}")
    print(f"\nNext steps:")
    print(f"  1. Update collect_revealed_preferences.py to use this model:")
    print(f"     model = '{finetuned_model_id}'")
    print(f"  2. Run collect_revealed_preferences.py to measure revealed preferences")
    print(f"  3. Compare measured ELOs to target priorities in {priorities_file}")
    print(f"\n{'='*80}\n")

else:
    print(f"\n{'='*80}")
    print(f"FINE-TUNING FAILED")
    print(f"{'='*80}\n")
