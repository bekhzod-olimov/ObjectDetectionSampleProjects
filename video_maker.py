import cv2
import os
import sys
from glob import glob

def images_to_video(image_folder, output_video_dir, n_ims, fps=30):
    # Get sorted image paths
    images = sorted(glob(os.path.join(image_folder, '*')))[:n_ims]
    if not images:
        print(f"No images found in: {image_folder}")
        return

    # Read first image to get dimensions
    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ds_name = image_folder.split("object_detection_project_datasets/")[-1].split("/")[0]    
    temp = os.path.join(output_video_dir, ds_name)
    os.makedirs(temp, exist_ok=True)    
    output_video_path = os.path.join(temp, f"{ds_name}.mp4")
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        if img.shape != (height, width, 3):
            img = cv2.resize(img, (width, height))        
        for _ in range(fps):
            video.write(img)

    video.release()
    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python video_maker.py <input_folder> <output_path> [fps=30]")
        print("Example: python video_maker.py ./images output.mp4 24")
    else:
        input_folder = sys.argv[1]
        output_path = sys.argv[2]
        fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        n_ims = int(sys.argv[4])
        print("Video is being generated...")
        images_to_video(input_folder, output_path, fps, n_ims)