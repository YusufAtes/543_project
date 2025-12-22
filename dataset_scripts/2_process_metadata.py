"""
Step 2: Process and merge metadata from artwork_dataset.csv and info_dataset.csv
Creates a unified metadata file with cleaned and merged information.
"""
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def process_metadata(
    artwork_csv="../../archive/artwork_dataset.csv",
    info_csv="../../archive/info_dataset.csv",
    image_mapping_file="../../dataset/image_mapping.json",
    output_csv="../../dataset/metadata.csv"
):
    """
    Load metadata CSVs, merge on artist name, and create unified metadata file.
    """
    print("Loading artwork dataset...")
    artwork_df = pd.read_csv(artwork_csv)
    print(f"Loaded {len(artwork_df)} artwork records")
    
    print("Loading info dataset...")
    info_df = pd.read_csv(info_csv)
    print(f"Loaded {len(info_df)} artist info records")
    
    # Load image mapping to get valid image IDs
    with open(image_mapping_file, 'r', encoding='utf-8') as f:
        image_mapping = json.load(f)
    
    valid_image_ids = set(image_mapping.keys())
    print(f"Valid image IDs: {len(valid_image_ids)}")
    
    # Clean artist names for merging (remove quotes, normalize)
    artwork_df['artist_clean'] = artwork_df['artist'].str.replace('"', '').str.strip()
    info_df['artist_clean'] = info_df['artist'].str.replace('"', '').str.strip()
    
    # Merge on artist name
    print("Merging metadata...")
    merged_df = artwork_df.merge(
        info_df,
        on='artist_clean',
        how='left',
        suffixes=('', '_info')
    )
    
    # Extract year from picture data if available
    def extract_year(picture_data):
        if pd.isna(picture_data):
            return None
        import re
        # Look for patterns like (1574-88) or c. 1596 or 1605-10
        matches = re.findall(r'(\d{4})', str(picture_data))
        if matches:
            return int(matches[0])
        return None
    
    merged_df['year'] = merged_df['picture data'].apply(extract_year)
    
    # Extract medium/material from picture data
    def extract_medium(picture_data):
        if pd.isna(picture_data):
            return None
        text = str(picture_data).lower()
        mediums = ['oil on canvas', 'oil on wood', 'oil on panel', 'oil on copper', 
                  'fresco', 'watercolor', 'tempera', 'charcoal', 'pencil', 
                  'marble', 'bronze', 'wood', 'canvas', 'panel', 'copper']
        for medium in mediums:
            if medium in text:
                return medium
        return None
    
    merged_df['medium'] = merged_df['picture data'].apply(extract_medium)
    
    # Extract filename from jpg_url to match with actual image files
    def extract_filename(url):
        if pd.isna(url):
            return None
        url_str = str(url)
        filename = url_str.split('/')[-1]
        return filename.replace('.jpg', '').replace('.JPG', '')
    
    merged_df['filename_base'] = merged_df['jpg url'].apply(extract_filename)
    merged_df['image_id'] = merged_df['filename_base']
    
    # Also keep the original ID column for reference
    merged_df['original_id'] = merged_df['ID'].astype(str)
    
    # Select and rename columns for final metadata
    metadata_df = merged_df[[
        'image_id', 'artist', 'title', 'year', 'medium', 
        'period', 'school', 'nationality', 'picture data', 
        'file info', 'jpg url'
    ]].copy()
    
    # Clean up column names
    metadata_df.columns = [
        'image_id', 'artist', 'title', 'year', 'medium',
        'period', 'school', 'nationality', 'picture_data',
        'file_info', 'jpg_url'
    ]
    
    # Fill missing values
    metadata_df = metadata_df.fillna({
        'period': 'Unknown',
        'school': 'Unknown',
        'nationality': 'Unknown',
        'year': 'Unknown',
        'medium': 'Unknown'
    })
    
    # Save metadata
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nSaved metadata to {output_path}")
    print(f"Total metadata records: {len(metadata_df)}")
    print(f"\nMetadata columns: {list(metadata_df.columns)}")
    print(f"\nSample record:")
    print(metadata_df.iloc[0].to_dict())
    
    return metadata_df

if __name__ == "__main__":
    df = process_metadata()

