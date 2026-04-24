import csv
import json
from pathlib import Path

def create_descriptions_jsonl(
    image_dir="cropped_artifacts",
    descriptions_csv="artifact_descriptions.csv",
    output_file="all_artifacts.json"
):
    """Convert CSV descriptions to JSONL format for CeramicArtifactDataset
    
    Args:
        image_dir: Directory containing cropped artifact images
        descriptions_csv: Path to CSV file with descriptions
        output_file: Output JSONL file path
    """
    
    image_dir = Path(image_dir)
    descriptions_csv = Path(descriptions_csv)
    
    # Check if files exist
    if not image_dir.exists():
        print(f"❌ Error: Image directory '{image_dir}' not found")
        return False
    
    if not descriptions_csv.exists():
        print(f"❌ Error: CSV file '{descriptions_csv}' not found")
        return False
    
    # Get all images in directory
    image_files = set()
    for img_path in image_dir.glob("*.png"):
        image_files.add(img_path.name)
    
    print(f"Found {len(image_files)} images in {image_dir}\n")
    
    # Read descriptions from CSV
    descriptions_data = {}
    with open(descriptions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            description = row.get('description', 'ceramic artifact')
            if filename:
                descriptions_data[filename] = description
    
    print(f"Read {len(descriptions_data)} descriptions from {descriptions_csv}\n")
    
    # Create JSONL file
    matched_count = 0
    unmatched_images = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_file in sorted(image_files):
            # Try to find matching description
            description = descriptions_data.get(
                image_file,
                "ceramic artifact with decorative patterns"
            )
            
            # Create JSON entry
            entry = {
                "filename": image_file,
                "image_path": str((image_dir / image_file).absolute()),
                "description": description
            }
            
            # Write to JSONL
            f.write(json.dumps(entry) + '\n')
            
            if image_file in descriptions_data:
                matched_count += 1
            else:
                unmatched_images.append(image_file)
    
    print(f"✓ Created {output_file}")
    print(f"  Matched descriptions: {matched_count}/{len(image_files)}")
    print(f"  Using default description: {len(unmatched_images)}")
    
    if unmatched_images and len(unmatched_images) <= 10:
        print(f"\n  Unmatched images:")
        for img in unmatched_images:
            print(f"    - {img}")
    
    return True

def create_comprehensive_csv(
    image_dir="cropped_artifacts",
    descriptions_csv="artifact_descriptions.csv",
    output_csv="all_artifacts.csv"
):
    """Create comprehensive CSV with all artifact information
    
    Args:
        image_dir: Directory containing cropped artifact images
        descriptions_csv: Path to CSV file with descriptions
        output_csv: Output CSV file path
    """
    
    image_dir = Path(image_dir)
    descriptions_csv = Path(descriptions_csv)
    
    # Load existing descriptions
    descriptions_data = {}
    with open(descriptions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            description = row.get('description', '')
            if filename:
                descriptions_data[filename] = description
    
    # Get all images
    image_files = sorted(image_dir.glob("*.png"))
    
    # Create comprehensive CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'id',
            'filename',
            'image_path',
            'description',
            'has_description',
            'file_size_mb',
            'status'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, image_path in enumerate(image_files, 1):
            filename = image_path.name
            description = descriptions_data.get(filename, "ceramic artifact")
            has_description = filename in descriptions_data
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            
            row = {
                'id': idx,
                'filename': filename,
                'image_path': str(image_path.absolute()),
                'description': description,
                'has_description': 'yes' if has_description else 'no',
                'file_size_mb': f"{file_size_mb:.2f}",
                'status': 'ready'
            }
            writer.writerow(row)
    
    print(f"✓ Created comprehensive CSV: {output_csv}")
    return True

def verify_dataset(
    image_dir="cropped_artifacts",
    descriptions_file="all_artifacts.json"
):
    """Verify that the dataset is correctly formatted
    
    Args:
        image_dir: Directory containing cropped artifact images
        descriptions_file: JSONL file with descriptions
    """
    
    image_dir = Path(image_dir)
    descriptions_file = Path(descriptions_file)
    
    print("Verifying dataset...\n")
    
    # Check image directory
    image_files = list(image_dir.glob("*.png"))
    print(f"✓ Found {len(image_files)} images in {image_dir}")
    
    # Check JSONL file
    if not descriptions_file.exists():
        print(f"❌ JSONL file not found: {descriptions_file}")
        return False
    
    entries = []
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON: {e}")
                return False
    
    print(f"✓ Found {len(entries)} entries in {descriptions_file}")
    
    # Verify all images have descriptions
    image_names = {img.name for img in image_files}
    described_names = {entry['filename'] for entry in entries}
    
    missing = image_names - described_names
    extra = described_names - image_names
    
    if missing:
        print(f"⚠ Missing descriptions for {len(missing)} images:")
        for name in sorted(list(missing)[:5]):
            print(f"  - {name}")
    
    if extra:
        print(f"⚠ Extra descriptions for {len(extra)} images not in directory")
    
    # Print sample entries
    print(f"\nSample entries:")
    for entry in entries[:3]:
        print(f"  - {entry['filename']}: {entry['description'][:60]}...")
    
    print(f"\n✓ Dataset verification complete!")
    return True

# Main execution
if __name__ == "__main__":
    import sys
    
    # Define paths
    image_dir = "data/cropped_artifacts"
    descriptions_csv = "data/artifact_descriptions.csv"
    output_jsonl = "data/all_artifacts.json"
    output_csv = "data/all_artifacts_comprehensive.csv"
    
    print("="*70)
    print("PREPARING DATASET FOR FINETUNING")
    print("="*70)
    
    # Step 1: Create JSONL from CSV
    print("\n[1/3] Creating JSONL file...")
    success = create_descriptions_jsonl(
        image_dir=image_dir,
        descriptions_csv=descriptions_csv,
        output_file=output_jsonl
    )
    
    if not success:
        sys.exit(1)
    
    # Step 2: Create comprehensive CSV
    print("\n[2/3] Creating comprehensive CSV...")
    create_comprehensive_csv(
        image_dir=image_dir,
        descriptions_csv=descriptions_csv,
        output_csv=output_csv
    )
    
    # Step 3: Verify dataset
    print("\n[3/3] Verifying dataset...")
    verify_dataset(
        image_dir=image_dir,
        descriptions_file=output_jsonl
    )
    
    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*70)
    print(f"\nYou can now use the following in your finetuning script:")
    print(f"  image_dir: '{image_dir}'")
    print(f"  descriptions_file: '{output_jsonl}'")
    print(f"\nRun finetuning with:")
    print(f"  uv run python scripts/modeling/finetune_vanilla_diffuser.py")