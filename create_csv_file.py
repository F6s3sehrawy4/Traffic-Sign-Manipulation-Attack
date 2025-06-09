import os
import csv

# Specify the path to your main folder containing subfolders of images
folder_path = "dataset/test"
output_csv = "sample_submission.csv"

# Open a CSV file to write
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header with the new "id" column
    writer.writerow(["ID", "ImagePath", "Label"])

    image_id = 1  # Start ID from 1 (or any number you prefer)

    # Walk through the folder structure
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # Check if the path is a directory (subfolder)
        if os.path.isdir(subfolder_path):
            # Iterate over each file in the subfolder
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)

                # Only process image files (filter extensions if needed)
                if os.path.isfile(image_path):
                    # Write the ID, image path, and the label (subfolder name) to the CSV
                    writer.writerow([image_id, image_path, subfolder_name])
                    image_id += 1  # Increment the ID for the next image

print(f"CSV file '{output_csv}' created successfully!")
