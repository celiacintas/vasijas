import argparse
from ultralytics import YOLO
import cv2
from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1, box2: tuples of (x1, y1, x2, y2)
    
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def remove_overlapping_boxes(boxes, iou_threshold=0.3):
    """Remove overlapping bounding boxes based on IoU threshold
    
    Args:
        boxes: List of dicts with bounding box info
        iou_threshold: Maximum allowed IoU between boxes (0-1)
    
    Returns:
        List of non-overlapping boxes
    """
    if not boxes:
        return boxes
    
    # Sort by confidence (descending) to keep better detections
    sorted_boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    kept_boxes = []
    
    for current_box in sorted_boxes:
        current_coords = (current_box['x1'], current_box['y1'], 
                         current_box['x2'], current_box['y2'])
        
        # Check overlap with already kept boxes
        overlaps = False
        for kept_box in kept_boxes:
            kept_coords = (kept_box['x1'], kept_box['y1'], 
                          kept_box['x2'], kept_box['y2'])
            
            iou = calculate_iou(current_coords, kept_coords)
            
            if iou > iou_threshold:
                overlaps = True
                break
        
        if not overlaps:
            kept_boxes.append(current_box)
    
    return kept_boxes

def detect_artifacts_segmentation(image_path, output_dir="cropped_artifacts", 
                                  min_width=20, min_height=20, min_area=2500,
                                  iou_threshold=0.3):
    """Use YOLOv8 Segmentation for better artifact detection
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for cropped artifacts
        min_width: Minimum bounding box width in pixels
        min_height: Minimum bounding box height in pixels
        min_area: Minimum bounding box area in pixels
        iou_threshold: Maximum allowed IoU to consider boxes as overlapping
    """
    
    # Load segmentation model
    model = YOLO('yolov8m-seg.pt')  # segmentation model
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    image_stem = Path(image_path).stem
    
    # Run inference
    results = model(image_path, conf=0.002)
    
    candidate_artifacts = []
    skipped = 0
    
    for idx, result in enumerate(results):
        if result.masks is not None:
            masks = result.masks.data
            boxes = result.boxes
            
            for mask_idx, (mask, box) in enumerate(zip(masks, boxes), 1):
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filter by size constraints
                if width < min_width or height < min_height or area < min_area:
                    skipped += 1
                    continue
                
                # Add to candidates
                candidate_artifacts.append({
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': int(width),
                    'height': int(height),
                    'area': int(area),
                    'confidence': float(conf)
                })
    
    # Remove overlapping boxes
    artifacts = remove_overlapping_boxes(candidate_artifacts, iou_threshold=iou_threshold)
    
    # Re-index after filtering
    for idx, artifact in enumerate(artifacts, 1):
        artifact['id'] = idx
        artifact['filename'] = f"{image_stem}_artifact_{idx:02d}.png"
        artifact['source_image'] = Path(image_path).name
    
    # Crop and save non-overlapping artifacts
    for artifact in artifacts:
        x1, y1, x2, y2 = artifact['x1'], artifact['y1'], artifact['x2'], artifact['y2']
        crop = image[y1:y2, x1:x2]
        
        filepath = output_path / artifact['filename']
        cv2.imwrite(str(filepath), crop)
    
    # Track how many were removed due to overlap
    overlap_removed = len(candidate_artifacts) - len(artifacts)
    
    return artifacts, skipped, overlap_removed

def process_all_images(input_dir="filtered_images", output_base_dir="cropped_artifacts",
                       min_width=50, min_height=50, min_area=2500, iou_threshold=0.3):
    """Process all images in input directory
    
    Args:
        input_dir: Directory containing filtered images
        output_base_dir: Base output directory for all artifacts
        min_width: Minimum bounding box width in pixels
        min_height: Minimum bounding box height in pixels
        min_area: Minimum bounding box area in pixels
        iou_threshold: Maximum allowed IoU to consider boxes as overlapping (0-1)
    """
    
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' not found")
        return None
    
    # Get all image files
    image_files = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    
    if not image_files:
        print(f"⚠ No images found in {input_dir}")
        return None
    
    print(f"Found {len(image_files)} images to process")
    print(f"IoU threshold for overlap removal: {iou_threshold}\n")
    
    # Global tracking
    all_artifacts = []
    total_skipped = 0
    total_overlap_removed = 0
    summary_data = []
    
    # Process each image
    for image_idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*70}")
        print(f"[{image_idx}/{len(image_files)}] Processing: {image_path.name}")
        print(f"{'='*70}")
        
        try:
            # Detect artifacts with overlap removal
            artifacts, skipped, overlap_removed = detect_artifacts_segmentation(
                str(image_path),
                str(output_base_path),
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
                iou_threshold=iou_threshold
            )
            
            # Track results
            total_skipped += skipped
            total_overlap_removed += overlap_removed
            
            if artifacts:
                print(f"✓ Detected {len(artifacts)} non-overlapping artifacts from {image_path.name}")
                
                for artifact in artifacts:
                    all_artifacts.append(artifact)
                    print(f"  - {artifact['filename']}: {artifact['width']}x{artifact['height']}px "
                          f"area={artifact['area']}px² conf={artifact['confidence']:.3f}")
                
                print(f"✓ Saved to {output_base_path}")
            else:
                print(f"⚠ No artifacts detected in {image_path.name}")
            
            if skipped > 0:
                print(f"⚠ Skipped {skipped} artifacts (too small)")
            
            if overlap_removed > 0:
                print(f"⚠ Removed {overlap_removed} overlapping artifacts (IoU > {iou_threshold})")
            
            summary_data.append({
                'source_image': image_path.name,
                'detected': len(artifacts),
                'skipped_size': skipped,
                'removed_overlap': overlap_removed
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            summary_data.append({
                'source_image': image_path.name,
                'detected': 0,
                'skipped_size': 0,
                'removed_overlap': 0,
                'error': str(e)
            })
    
    # Save global CSV with all artifacts
    if all_artifacts:
        global_csv = output_base_path / "all_artifacts.csv"
        with open(global_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_artifacts[0].keys())
            writer.writeheader()
            writer.writerows(all_artifacts)
        print(f"\n✓ Global artifacts CSV saved to {global_csv}")
    
    # Save summary
    summary_csv = output_base_path / "processing_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"✓ Processing summary saved to {summary_csv}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total artifacts detected: {len(all_artifacts)}")
    print(f"Total artifacts skipped (too small): {total_skipped}")
    print(f"Total artifacts removed (overlapping): {total_overlap_removed}")
    print(f"IoU threshold used: {iou_threshold}")
    print(f"Output directory: {output_base_path.absolute()}")
    print(f"{'='*70}\n")
    
    return all_artifacts, summary_data

def main():
    parser = argparse.ArgumentParser(
        description="Extract individual artifacts from images using YOLOv8 segmentation with overlap removal."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="filtered_images",
        help="Input directory containing images (default: filtered_images)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="cropped_artifacts",
        help="Output directory for cropped artifacts (default: cropped_artifacts)"
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for overlap removal (0-1, default: 0.3). Higher = more permissive, lower = stricter"
    )
    
    parser.add_argument(
        "--min-width",
        type=int,
        default=50,
        help="Minimum bounding box width in pixels (default: 50)"
    )
    
    parser.add_argument(
        "--min-height",
        type=int,
        default=50,
        help="Minimum bounding box height in pixels (default: 50)"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=2500,
        help="Minimum bounding box area in pixels (default: 2500)"
    )
    
    args = parser.parse_args()
    
    # Validate IoU threshold
    if not (0 <= args.iou_threshold <= 1):
        print(f"❌ Error: IoU threshold must be between 0 and 1, got {args.iou_threshold}")
        return
    
    # Validate input directory exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input directory '{args.input}' does not exist")
        return
    
    print(f"Configuration:")
    print(f"  Input directory: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Min width: {args.min_width}px")
    print(f"  Min height: {args.min_height}px")
    print(f"  Min area: {args.min_area}px²\n")
    
    # Process all images
    all_artifacts, summary = process_all_images(
        input_dir=args.input,
        output_base_dir=args.output,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
        iou_threshold=args.iou_threshold
    )
    
    # Print summary statistics
    if all_artifacts:
        print("\nDetected artifacts by size:")
        sizes = {}
        for artifact in all_artifacts:
            size_range = f"{artifact['width']}x{artifact['height']}"
            sizes[size_range] = sizes.get(size_range, 0) + 1
        
        for size, count in sorted(sizes.items()):
            print(f"  {size}px: {count} artifacts")

if __name__ == "__main__":
    main()