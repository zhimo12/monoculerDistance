import os
import glob

# 1. Setup paths
data_path = r"D:\DepthTrain\kittidata"
split_path = os.path.join("splits", "custom", "test_files.txt")

search_path = os.path.join(data_path, "*_sync")
drives = glob.glob(search_path)

print(f"Found {len(drives)} drive folders.")

with open(split_path, "w") as f:
    count = 0
    for drive in drives:
        drive_name = os.path.basename(drive)
        
        # Define Image and Velodyne folders
        image_folder = os.path.join(drive, "image_02", "data")
        velo_folder = os.path.join(drive, "velodyne_points", "data")
        
        if not os.path.exists(image_folder) or not os.path.exists(velo_folder):
            print(f"Skipping {drive_name} (Missing images or velodyne folder)")
            continue
            
        images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        if not images:
            images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
            
        for i, img_path in enumerate(images):
            # We take every 5th frame to speed it up
            if i % 5 == 0:
                frame_str = os.path.splitext(os.path.basename(img_path))[0]
                frame_idx = int(frame_str)
                
                # --- CRITICAL CHECK: Does the Velodyne file exist? ---
                velo_filename = f"{frame_idx:010d}.bin"
                velo_path = os.path.join(velo_folder, velo_filename)
                
                if os.path.exists(velo_path):
                    # Only add if we have BOTH image and depth data
                    line = f"{drive_name} {frame_idx} l\n"
                    f.write(line)
                    count += 1
                # -----------------------------------------------------

print(f"SUCCESS: Created VALID custom split with {count} pairs.")
print(f"Saved to: {split_path}")