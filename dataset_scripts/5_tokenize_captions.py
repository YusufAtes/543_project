"""
Step 5: Tokenize captions using GPT-2 tokenizer from HuggingFace
Saves tokenized captions with input_ids and attention_mask.
"""
import json
from pathlib import Path
from transformers import GPT2Tokenizer
from tqdm import tqdm

def tokenize_captions(
    captions_file="../../dataset/captions.json",
    id_mapping_file="../../dataset/id_mapping.json",
    output_file="../../dataset/tokenized_captions.json",
    max_length=256
):
    """
    Tokenize all captions using GPT-2 tokenizer.
    Maps old image IDs to new standardized IDs.
    """
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading captions...")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print("Loading ID mapping...")
    with open(id_mapping_file, 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    
    # Reverse mapping: new_id -> old_id
    reverse_mapping = {v: k for k, v in id_mapping.items()}
    
    print(f"Tokenizing {len(captions)} captions...")
    
    tokenized_captions = {}
    
    for new_id, old_id in tqdm(reverse_mapping.items(), desc="Tokenizing captions"):
        if old_id not in captions:
            continue
        
        caption_text = captions[old_id]
        
        # Tokenize with max_length, truncation, and padding
        encoded = tokenizer(
            caption_text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'  # Return as numpy arrays for easy JSON serialization
        )
        
        tokenized_captions[new_id] = {
            'input_ids': encoded['input_ids'][0].tolist(),
            'attention_mask': encoded['attention_mask'][0].tolist()
        }
    
    # Save tokenized captions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenized_captions, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(tokenized_captions)} tokenized captions to {output_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Print example
    if tokenized_captions:
        example_id = list(tokenized_captions.keys())[0]
        example = tokenized_captions[example_id]
        print(f"\nExample tokenization (ID: {example_id}):")
        print(f"  Input IDs length: {len(example['input_ids'])}")
        print(f"  Attention mask sum: {sum(example['attention_mask'])}")
        print(f"  Decoded back: {tokenizer.decode(example['input_ids'][:50], skip_special_tokens=True)[:100]}...")
    
    return tokenized_captions

if __name__ == "__main__":
    tokenized = tokenize_captions()

