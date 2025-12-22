"""
Step 6: Create train/val/test splits (80/10/10) and save split files
Ensures balanced distribution if possible.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

def create_splits(
    tokenized_captions_file="../../dataset/tokenized_captions.json",
    metadata_csv="../../dataset/metadata.csv",
    id_mapping_file="../../dataset/id_mapping.json",
    output_dir="../../dataset/splits/",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """
    Create train/val/test splits ensuring balanced distribution.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    print("Loading tokenized captions...")
    with open(tokenized_captions_file, 'r', encoding='utf-8') as f:
        tokenized_captions = json.load(f)
    
    all_image_ids = sorted(tokenized_captions.keys())
    print(f"Total images: {len(all_image_ids)}")
    
    # Load metadata for potential stratification
    try:
        import pandas as pd
        metadata_df = pd.read_csv(metadata_csv)
        
        # Load ID mapping to get artist info for stratification
        with open(id_mapping_file, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        
        # Create artist mapping
        metadata_df['image_id_str'] = metadata_df['image_id'].astype(str)
        artist_map = {}
        for _, row in metadata_df.iterrows():
            old_id = str(row['image_id'])
            if old_id in reverse_mapping:
                new_id = reverse_mapping[old_id]
                artist_map[new_id] = row.get('artist', 'Unknown')
        
        # Group by artist for stratification
        artist_groups = defaultdict(list)
        for img_id in all_image_ids:
            artist = artist_map.get(img_id, 'Unknown')
            artist_groups[artist].append(img_id)
        
        print(f"Found {len(artist_groups)} artists for stratification")
        
        # Stratified split
        train_ids = []
        val_ids = []
        test_ids = []
        
        random.seed(seed)
        
        for artist, img_list in artist_groups.items():
            random.shuffle(img_list)
            n = len(img_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_ids.extend(img_list[:n_train])
            val_ids.extend(img_list[n_train:n_train+n_val])
            test_ids.extend(img_list[n_train+n_val:])
        
        # Shuffle the final lists
        random.shuffle(train_ids)
        random.shuffle(val_ids)
        random.shuffle(test_ids)
        
    except Exception as e:
        print(f"Warning: Could not stratify by artist ({e}). Using random split.")
        # Fallback to random split
        random.seed(seed)
        random.shuffle(all_image_ids)
        
        n = len(all_image_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_ids = all_image_ids[:n_train]
        val_ids = all_image_ids[n_train:n_train+n_val]
        test_ids = all_image_ids[n_train+n_val:]
    
    # Save split files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "train_ids.txt"
    val_file = output_path / "val_ids.txt"
    test_file = output_path / "test_ids.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_ids))
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_ids))
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_ids))
    
    print(f"\nSplit created:")
    print(f"  Train: {len(train_ids)} images ({100*len(train_ids)/len(all_image_ids):.1f}%)")
    print(f"  Val:   {len(val_ids)} images ({100*len(val_ids)/len(all_image_ids):.1f}%)")
    print(f"  Test:  {len(test_ids)} images ({100*len(test_ids)/len(all_image_ids):.1f}%)")
    print(f"\nSaved to:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")
    
    return train_ids, val_ids, test_ids

if __name__ == "__main__":
    train, val, test = create_splits()

