import cv2
import os
import sys
from glob import glob

def images_to_video(image_folder, output_video_path, fps=30):
    # Get sorted image paths
    images = sorted(glob(os.path.join(image_folder, '*')))
    if not images:
        print(f"No images found in: {image_folder}")
        return

    # Read first image to get dimensions
    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        print("Usage: python img2video.py <input_folder> <output_path> [fps=30]")
        print("Example: python img2video.py ./images output.mp4 24")
    else:
        input_folder = sys.argv[1]
        output_path = sys.argv[2]
        fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        images_to_video(input_folder, output_path, fps)