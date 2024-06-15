import os
import sys
from image_helper import dataset_comp
import piece_maker

def main():
    # Get folder path and size from command line arguments
    folder_path, size = sys.argv[1], int(sys.argv[2])
    print(f"Processing images in folder: {folder_path} with size: {size}")

    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Construct the full path to the image
            gt_path = os.path.join(folder_path, filename)
            print(f"Processing image: {gt_path}")

            # Process the image
            piece_maker.main(gt_path, size, size)
            img_name, _ = os.path.splitext(filename)

            # Call dataset_comp for the processed image
            dataset_comp(img_name, size)


if __name__ == '__main__':
    main()