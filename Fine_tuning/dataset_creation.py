import pandas as pd
import os
from tqdm import tqdm
from pydub import AudioSegment
import glob
import re
import csv
import json

# --- Configuration ---

DATASET_DIR = '../cv-corpus'  # Path to the dataset directory

manifest_validated_mp3 = os.path.join(DATASET_DIR, "validated_files.tsv") # Path to the TSV file with validated files
manifest_validated_wav = os.path.join(DATASET_DIR, "manifest_validated.jsonl") # Path to save the WAV manifest

duration_file = os.path.join(DATASET_DIR, "durations.tsv") # Path to save durations

wav_clips_dir = os.path.join(DATASET_DIR, "wav_clips") # Directory to save converted WAV files
mp3_clips_dir = os.path.join(DATASET_DIR, "it/clips") # Directory containing original MP3 files

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\…\{\}\(\)\[\]\–\’\'\/]' # Characters to remove from text

split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}

train_manifest = os.path.join(DATASET_DIR, "train_manifest.jsonl")
validation_manifest = os.path.join(DATASET_DIR, "validation_manifest.jsonl")
test_manifest = os.path.join(DATASET_DIR, "test_manifest.jsonl")


# --- Functions ---

def count_lines(filepath):
    """
    Counts the number of lines in a file
    """
    count = 0

    with open(filepath, 'r', encoding='utf-8') as file:
        for _ in file:
            count += 1
    return count - 1

def MCV_to_Nemo(sample_rate=16000, mono=True, delete_mp3=False):
    """
    This function:
      - Converts MP3 files listed in a TSV manifest to WAV format
      - Creates a new WAV manifest .jsonl containing the 'text', 'audio_filepath' and 'duration' fields.
    """

    if not os.path.exists(wav_clips_dir):
        os.makedirs(wav_clips_dir)

    try:
        num_rows = count_lines(manifest_validated_mp3)

        with open(manifest_validated_mp3, mode='r', encoding='utf-8') as read_file:

            reader = csv.DictReader(read_file, delimiter="\t")
            original_fieldnames = reader.fieldnames

            if 'path' not in original_fieldnames or 'sentence' not in original_fieldnames:
                raise ValueError("Columns 'path' and 'sentence' must be present in the TSV file.")

            # Open the output file for writing in JSONL format
            with open(manifest_validated_wav, mode='w', encoding='utf-8') as write_file:

                with tqdm(total=num_rows, desc="Converting MP3 to WAV and creating JSONL manifest") as pbar:
                    for line in reader:

                        mp3_file_name = line['path']

                        mp3_path = os.path.join(mp3_clips_dir, mp3_file_name)
                        wav_file_name = mp3_file_name.replace('.mp3', '.wav')

                        wav_path = os.path.join(wav_clips_dir, wav_file_name)

                        # Load, convert and export WAV file
                        audio = AudioSegment.from_mp3(mp3_path)

                        if mono:
                            audio = audio.set_channels(1)
                        audio = audio.set_frame_rate(sample_rate)

                        audio.export(wav_path, format="wav")

                        # The path in the manifest should be relative to the dataset root or absolute,
                        # depending on NeMo's requirement. Assuming relative to dataset root here.
                        # NeMo generally expects the 'audio_filepath' (renamed 'path' here) to be relative 
                        # to where the manifest is processed, or an absolute path.
                        
                        relative_wav_path = os.path.join(os.path.basename(wav_clips_dir), wav_file_name)


                        duration = len(audio) / 1000

                        output_line = {
                            # Using relative path for better portability, adjust if absolute path is needed
                            'text': line['sentence'], # NeMo convention uses 'text' for transcript
                            'audio_filepath': relative_wav_path,
                            'duration': duration
                        }

                        # Write the JSON object followed by a newline (JSONL format)
                        write_file.write(json.dumps(output_line) + '\n')

                        # Delete original MP3 file if requested
                        if delete_mp3:
                            os.remove(mp3_path)

                        pbar.update(1)

        print("Conversion and manifest creation completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")


def remove_mp3(clips_dir):
    files = glob.glob(os.path.join(clips_dir, "*.mp3"))
    for file in files:
        os.remove(file)


def process_and_normalize_manifest(json_file):
    
    if not os.path.exists(json_file):
        print(f"Errore: File not found{json_file}")
        return

    df = pd.read_json(json_file, lines=True)

    # Allowed characters regex: finds everything that is NOT a-z, accented letters, apostrophe, or space
    allowed_chars_regex = re.compile(r"[^a-zàèéìíîòóùú' ]")

    def clean_text(text):
        text = text.lower().strip()
        text = allowed_chars_regex.sub("", text)
        
        return text

    df["text"] = df["text"].apply(clean_text)

    df.to_json(json_file, orient="records", lines=True, force_ascii=False)
    print(f"Normalized JSON file saved to: {json_file}")


def create_dataset_splits(input_manifest, splits_ratios, train_out, validation_out, test_out):
    """
    Reads the input JSONL manifest, shuffles the data, and splits it into
    training, validation, and test sets based on the specified ratios.

    Args:
        input_manifest (str): Path to the single input manifest file (.jsonl).
        splits_ratios (dict): Dictionary with keys 'train', 'validation', 'test'
                              and float values summing to 1.0.
        train_out (str): Path to save the training manifest (.jsonl).
        validation_out (str): Path to save the validation manifest (.jsonl).
        test_out (str): Path to save the test manifest (.jsonl).
    """
    
    # 1. Load Data
    try:
        df = pd.read_json(input_manifest, lines=True)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {input_manifest}")
        return
    except ValueError as e:
        print(f"Error reading manifest {input_manifest}. Check the JSONL format: {e}")
        return

    # 2. Shuffle Data (Crucial for ensuring random distribution)
    # Use a fixed random_state for reproducibility
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total_size = len(df)
    
    # 3. Calculate split points
    train_size = int(splits_ratios['train'] * total_size)
    validation_size = int(splits_ratios['validation'] * total_size)
    # Test set takes the remainder to handle rounding issues

    # 4. Perform the split
    train_df = df.iloc[:train_size]
    validation_df = df.iloc[train_size : train_size + validation_size]
    test_df = df.iloc[train_size + validation_size :]
    
    print("-" * 50)
    print(f"Total size: {total_size}")
    print(f"Train Set: {len(train_df)} ({len(train_df)/total_size:.2%})")
    print(f"Validation Set: {len(validation_df)} ({len(validation_df)/total_size:.2%})")
    print(f"Test Set: {len(test_df)} ({len(test_df)/total_size:.2%})")
    print("-" * 50)

    # 5. Save the new manifests in JSONL format
    try:
        train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
        validation_df.to_json(validation_out, orient="records", lines=True, force_ascii=False)
        test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)
        print("Dataset split creation completed successfully.")
    except Exception as e:
        print(f"Error saving manifests: {e}")


if __name__ == "__main__":

    # Convert .mp3 to .wav for validated files only (delte after conversion)
    MCV_to_Nemo(delete_mp3=False)

    # Remove any remaining .mp3 files in the clips directory
    remove_mp3(mp3_clips_dir)

    # Preprocess text
    process_and_normalize_manifest(manifest_validated_wav)

    # Create the splits and generate manifests
    create_dataset_splits(
        manifest_validated_wav,
        split_ratios,
        train_manifest,
        validation_manifest,
        test_manifest
    )

    