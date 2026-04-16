from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shutil

# Load images
image_dir = Path("output_images")
image_paths = sorted(image_dir.glob("*picture*.png"))

images = []
image_arrays = []
image_names = []
target_size = (512, 512)

for img_path in image_paths:
    try:
        img = Image.open(img_path)
        # Resize to 512x512
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Store resized image for display
        images.append(img_resized.copy())
        
        # Convert to grayscale and flatten for UMAP
        img_gray = img_resized.convert('L')
        img_array = np.array(img_gray).flatten()
        image_arrays.append(img_array)
        image_names.append(img_path.name)
        print(f"Loaded and resized: {img_path.name}")
    except Exception as e:
        print(f"Error: {e}")

# Prepare data
X = np.array(image_arrays)
print(f"Data shape: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create UMAP
print("Computing UMAP...")
reducer = umap.UMAP(n_components=2,
            n_neighbors=20,
            min_dist=0.001,
            metric='euclidean',
            random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Create filtered images folder
filtered_dir = Path("filtered_images")
filtered_dir.mkdir(exist_ok=True)

# Filter based on UMAP coordinates
umap1_threshold = -7
umap2_threshold = 2

filtered_indices = []
filtered_images = []
filtered_names = []

for i, (umap1, umap2) in enumerate(X_umap):
    if umap1 > umap1_threshold and umap2 > umap2_threshold:
        filtered_indices.append(i)
        filtered_images.append(images[i])
        filtered_names.append(image_names[i])
        
        # Copy original image to filtered folder
        original_path = image_dir / image_names[i]
        destination = filtered_dir / image_names[i]
        #shutil.copy2(original_path, destination)
        
        print(f"✓ Filtered: {image_names[i]} (UMAP1={umap1:.2f}, UMAP2={umap2:.2f})")

print(f"\nFiltered {len(filtered_indices)} images out of {len(image_names)}")
print(f"Saved to: {filtered_dir.absolute()}\n")

# Plot all points with filtered ones highlighted
fig, ax = plt.subplots(figsize=(16, 12))

# Plot all points
ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.3, s=200, label="All images", color='gray')

# Highlight filtered points
filtered_umap = X_umap[filtered_indices]
ax.scatter(filtered_umap[:, 0], filtered_umap[:, 1], alpha=0.8, s=300, 
           label="Filtered images", color='red', edgecolors='darkred', linewidth=2)

# Add filter region rectangle
rect_x = [-7, 20, 20, -7, -7]
rect_y = [2, 2, 15, 15, 2]
ax.plot(rect_x, rect_y, 'r--', linewidth=2, label=f"Filter region (UMAP1 > {umap1_threshold}, UMAP2 > {umap2_threshold})")

# Add thumbnail images for filtered ones
for i, img in enumerate(filtered_images):
    # Resize for thumbnail display
    img_thumb = img.copy()
    img_thumb.thumbnail((60, 60))
    
    # Create OffsetImage
    imagebox = OffsetImage(img_thumb, zoom=1, alpha=0.9)
    
    # Annotate with image (use filtered_indices to get correct UMAP coordinates)
    umap_idx = filtered_indices[i]
    ab = AnnotationBbox(imagebox, (X_umap[umap_idx, 0], X_umap[umap_idx, 1]), 
                        frameon=True, pad=0.1, alpha=0.3)
    ax.add_artist(ab)

ax.set_title(f"UMAP Visualization - Filtered Images (UMAP1 > {umap1_threshold}, UMAP2 > {umap2_threshold})", 
             fontsize=16)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig("umap_visualization_filtered.png", dpi=300, bbox_inches='tight')
plt.show()

print("UMAP plot with filter region saved!")

# Save filter metadata
import json
metadata = {
    "total_images": len(image_names),
    "filtered_images": len(filtered_indices),
    "filter_criteria": {
        "umap1_greater_than": umap1_threshold,
        "umap2_greater_than": umap2_threshold
    },
    "filtered_image_names": filtered_names,
    "filtered_coordinates": [
        {
            "name": filtered_names[i],
            "umap1": float(X_umap[filtered_indices[i], 0]),
            "umap2": float(X_umap[filtered_indices[i], 1])
        }
        for i in range(len(filtered_names))
    ]
}

with open(filtered_dir / "filter_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved to: {filtered_dir / 'filter_metadata.json'}")