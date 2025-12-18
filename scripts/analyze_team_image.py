import os
import sys

# Try to import PIL, if not available, use alternatives
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not installed, using file properties only")

# Check file existence
image_path = r'd:\Anomaly dataset\Team extraction\undefined.png'

print("="*70)
print("TEAM IMAGE ANALYSIS")
print("="*70)

if os.path.exists(image_path):
    print(f"\n✓ File found: {image_path}")
    
    # Get file size
    file_size_bytes = os.path.getsize(image_path)
    print(f"File size: {file_size_bytes:,} bytes ({file_size_bytes/1024:.1f} KB)")
    
    if HAS_PIL:
        try:
            img = Image.open(image_path)
            
            print(f"\n=== IMAGE PROPERTIES ===")
            print(f"Format: {img.format}")
            print(f"Dimensions: {img.size[0]} × {img.size[1]} pixels")
            print(f"Aspect ratio: {img.size[0]/img.size[1]:.2f}:1")
            print(f"Color Mode: {img.mode}")
            print(f"DPI: {img.info.get('dpi', 'Not specified')}")
            print(f"Total pixels: {img.size[0] * img.size[1]:,}")
            
            # Get color statistics
            if img.mode == 'RGB':
                r, g, b = img.split()
                print(f"\nColor channel ranges:")
                print(f"  Red:   min={min(r.getdata())}, max={max(r.getdata())}")
                print(f"  Green: min={min(g.getdata())}, max={max(g.getdata())}")
                print(f"  Blue:  min={min(b.getdata())}, max={max(b.getdata())}")
            elif img.mode == 'RGBA':
                print(f"  Mode: RGB with Alpha channel")
        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        print("\nNote: PIL not available. File properties:")
        print(f"  - Size: {file_size_bytes:,} bytes")
        print(f"  - Can be loaded with PIL to get detailed properties")
else:
    print(f"\n✗ File NOT found at: {image_path}")
    
    # Check if Team extraction folder exists
    team_folder = r'd:\Anomaly dataset\Team extraction'
    if os.path.exists(team_folder):
        files = os.listdir(team_folder)
        print(f"\nFiles in Team extraction folder: {files}")
    else:
        print(f"\nTeam extraction folder not found")
        
        # List workspace structure
        workspace_path = r'd:\Anomaly dataset'
        if os.path.exists(workspace_path):
            print(f"\nContents of {workspace_path}:")
            for item in os.listdir(workspace_path):
                path = os.path.join(workspace_path, item)
                item_type = "DIR" if os.path.isdir(path) else "FILE"
                print(f"  [{item_type}] {item}")
