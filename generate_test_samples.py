import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def create_benign_nodule(size=(128, 128)):
    """Create a synthetic benign lung nodule"""
    image = np.zeros(size, dtype=np.float32)
    
    # Create circular/oval shape
    center_x, center_y = np.random.randint(40, 88, 2)
    radius_x = np.random.randint(15, 25)
    radius_y = np.random.randint(15, 25)
    
    for i in range(size[0]):
        for j in range(size[1]):
            distance = ((i - center_x) / radius_x) ** 2 + ((j - center_y) / radius_y) ** 2
            if distance <= 1:
                intensity = 1.0 - distance * 0.7
                # Add smooth texture
                intensity += np.random.normal(0, 0.05)
                image[i, j] = max(0, min(1, intensity))
    
    # Add mild noise
    image += np.random.normal(0, 0.03, size)
    image = np.clip(image, 0, 1)
    
    return image

def create_malignant_nodule(size=(128, 128)):
    """Create a synthetic malignant lung nodule"""
    image = np.zeros(size, dtype=np.float32)
    
    # Create irregular shape
    center_x, center_y = np.random.randint(40, 88, 2)
    base_radius = np.random.randint(12, 20)
    
    for i in range(size[0]):
        for j in range(size[1]):
            base_distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            
            # Irregular boundary with spiculations
            irregular_radius = base_radius * (1 + 0.4 * np.sin(8 * np.arctan2(j - center_y, i - center_x)))
            
            if base_distance <= irregular_radius:
                intensity = 1.0 - (base_distance / irregular_radius) * 0.5
                # Add rough texture
                intensity += np.random.normal(0, 0.1)
                image[i, j] = max(0, min(1, intensity))
    
    # Add spiculations (spiky extensions)
    num_spicules = np.random.randint(4, 8)
    for _ in range(num_spicules):
        angle = np.random.random() * 2 * np.pi
        length = np.random.randint(8, 15)
        width = np.random.randint(1, 3)
        
        for l in range(length):
            for w in range(-width, width + 1):
                x = int(center_x + (base_radius + l) * np.cos(angle) + w * np.sin(angle))
                y = int(center_y + (base_radius + l) * np.sin(angle) + w * np.cos(angle))
                
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    intensity = 1.0 - (l / length) * 0.8
                    image[x, y] = max(image[x, y], intensity)
    
    # Add more noise for texture
    image += np.random.normal(0, 0.08, size)
    image = np.clip(image, 0, 1)
    
    return image

def create_lung_background(size=(128, 128)):
    """Create lung tissue background"""
    # Create lung-like texture with vessels
    background = np.random.normal(0.3, 0.1, size)
    
    # Add some vessel-like structures
    for _ in range(5):
        start_x, start_y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        length = np.random.randint(20, 50)
        angle = np.random.random() * 2 * np.pi
        
        for i in range(length):
            x = int(start_x + i * np.cos(angle))
            y = int(start_y + i * np.sin(angle))
            
            if 0 <= x < size[0] and 0 <= y < size[1]:
                # Create vessel
                for w in range(-2, 3):
                    for h in range(-2, 3):
                        nx, ny = x + w, y + h
                        if 0 <= nx < size[0] and 0 <= ny < size[1]:
                            dist = np.sqrt(w**2 + h**2)
                            if dist <= 2:
                                background[nx, ny] += 0.2 * (1 - dist/2)
    
    return np.clip(background, 0, 1)

def generate_test_samples(num_samples=20, output_dir="test_samples"):
    """Generate random test samples for lung cancer classification"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"🎨 Generating {num_samples} test samples...")
    
    for i in range(num_samples):
        # Randomly choose between benign and malignant
        if np.random.random() > 0.5:
            nodule_type = "benign"
            nodule_image = create_benign_nodule()
        else:
            nodule_type = "malignant" 
            nodule_image = create_malignant_nodule()
        
        # Create lung background
        background = create_lung_background()
        
        # Combine nodule with background
        final_image = background.copy()
        nodule_mask = nodule_image > 0.1
        final_image[nodule_mask] = nodule_image[nodule_mask]
        
        # Convert to 8-bit for saving
        final_image_8bit = (final_image * 255).astype(np.uint8)
        
        # Save image
        filename = f"{nodule_type}_sample_{i+1:02d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, final_image_8bit)
        
        print(f"✅ Saved: {filename}")
    
    print(f"\n🎯 Generated {num_samples} test samples in '{output_dir}' directory")
    
    # Create a preview image
    create_preview_grid(output_dir)

def create_preview_grid(output_dir):
    """Create a grid preview of all generated samples"""
    import glob
    
    image_files = glob.glob(os.path.join(output_dir, "*.png"))
    image_files.sort()
    
    if not image_files:
        return
    
    # Read first few images to determine layout
    sample_images = []
    for file in image_files[:12]:  # Show first 12
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        sample_images.append(img)
    
    # Create grid
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (img, file) in enumerate(zip(sample_images, image_files[:12])):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(os.path.basename(file))
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_preview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Created sample preview: sample_preview.png")

if __name__ == "__main__":
    # Generate test samples
    generate_test_samples(num_samples=20)
    
    print("\n🎉 Test samples generated successfully!")
    print("📁 Use these images to test your Flask application:")
    print("   python app.py")
    print("   Then upload the generated PNG files")