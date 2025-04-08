import os
import re

def generate_custom_labels(image_dir="fruits-360/custom_images", output_file="custom_labels.txt"):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    with open(output_file, "w") as f:
        for filename in image_files:
            # Extract label: get first part before a number or underscore
            name_part = os.path.splitext(filename)[0]  # remove extension
            label_match = re.match(r"([A-Za-z\s]+)", name_part)
            label = label_match.group(1).strip().capitalize() if label_match else "Unknown"

            f.write(f"{filename},{label}\n")

    print(f"custom_labels.txt generated with {len(image_files)} entries.")

if __name__ == "__main__":
    generate_custom_labels()
