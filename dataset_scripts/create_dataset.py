"""
Main orchestrator script to create the complete dataset.
Runs all steps in sequence.
"""
import sys
import subprocess
from pathlib import Path
import os

def main():
    """
    Run all dataset creation steps in sequence.
    """
    print("="*60)
    print("Dataset Creation Pipeline")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    base_dir = scripts_dir.parent.parent
    
    # Change to scripts directory for relative paths to work
    os.chdir(scripts_dir)
    
    steps = [
        ("Step 1: Load and validate images", "1_load_images.py"),
        ("Step 2: Process and merge metadata", "2_process_metadata.py"),
        ("Step 3: Generate long-form captions", "3_generate_captions.py"),
        ("Step 4: Preprocess images", "4_preprocess_images.py"),
        ("Step 5: Tokenize captions", "5_tokenize_captions.py"),
        ("Step 6: Create train/val/test splits", "6_create_splits.py"),
    ]
    
    for step_name, script_name in steps:
        print(f"\n{'='*60}")
        print(step_name)
        print('='*60)
        
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"Error: Script {script_path} not found!")
            continue
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=False,
                cwd=str(scripts_dir)
            )
            if result.returncode != 0:
                print(f"Warning: {step_name} exited with code {result.returncode}")
                response = input("\nContinue to next step? (y/n): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
        except Exception as e:
            print(f"Error in {step_name}: {e}")
            import traceback
            traceback.print_exc()
            response = input("\nContinue to next step? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    print("\n" + "="*60)
    print("Dataset creation complete!")
    print("="*60)
    print("\nFinal dataset structure:")
    print("dataset/")
    print("  images/              # Preprocessed 128x128 images")
    print("  captions.json        # Long-form captions (100-200 words)")
    print("  metadata.csv         # Cleaned metadata")
    print("  tokenized_captions.json  # Tokenized captions")
    print("  id_mapping.json      # Old ID -> New ID mapping")
    print("  splits/")
    print("    train_ids.txt")
    print("    val_ids.txt")
    print("    test_ids.txt")

if __name__ == "__main__":
    main()

