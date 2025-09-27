import pandas as pd
import os
from pathlib import Path

ROOT = r"C:\Users\20231807\Documents\GitHub\Cap_coral_reefs\New_version_of_the_project\website\benthic_datasets\mask_labels\reef_support"

def creating_dataframe():
    df = pd.DataFrame(columns=["type", "name", "name_ext", "path"])

    dirlist = os.listdir(ROOT)

    for dir in dirlist:
        new_address = os.path.join(ROOT, dir)
        if os.path.exists(new_address):
            print(f"There is dir {new_address}")
            images_path = os.path.join(new_address, "images")
            masks_stitched_path = os.path.join(new_address, "masks_stitched")
            labels_path = os.path.join(new_address,"labels")
            print(f"Lets check: images path -> {images_path} and masks_stitched path {masks_stitched_path}")
            print("\n")

            # Convert to Path objects
            images_path = Path(images_path)
            masks_stitched_path = Path(masks_stitched_path)

            # Iterate over masks in dir
            if masks_stitched_path.exists():  # Check if directory exists
                for mask in masks_stitched_path.iterdir():
                    if mask.is_file():
                        mask_name_ext = mask.name
                        print(mask.name)
                        mask_name = mask.stem
                        print(mask.stem)
                        mask_path = str(mask)  # Convert to string for DataFrame
                        print(str(mask))

                        # Create new row
                        new_row_mask = pd.DataFrame({
                            "type": ["mask"],  # Note: lists for DataFrame creation
                            "name": [mask_name],
                            "name_ext": [mask_name_ext],
                            "path": [mask_path]
                        })

                        df = pd.concat([df, new_row_mask], ignore_index=True)

            # Iterate over images in dir
            if images_path.exists():  # Check if directory exists
                for image in images_path.iterdir():
                    if image.is_file():
                        image_name_ext = image.name
                        print(image.name)
                        image_name = image.stem  # FIXED: was image.name
                        print(image.stem)
                        image_path = str(image)  # Convert to string for DataFrame
                        print(str(image))

                        # Create new row
                        new_row_image = pd.DataFrame({
                            "type": ["image"],  # Note: lists for DataFrame creation
                            "name": [image_name],
                            "name_ext": [image_name_ext],
                            "path": [image_path]
                        })

                        df = pd.concat([df, new_row_image], ignore_index=True)


    df.to_csv("output.csv",index=False)

    print(df)


def adding_labels():
    df = pd.read_csv("output.csv")

    dirlist = os.listdir(ROOT)

    for dir in dirlist:
        new_address = os.path.join(ROOT, dir)
        if os.path.exists(new_address):
            print(f"There is dir {new_address}")
            labels_path = Path(os.path.join(new_address, "labels"))  # Convert to Path
            print("\n")

            if labels_path.exists():
                for text in labels_path.iterdir():
                    if text.is_file():
                        text_name_ext = text.name
                        text_name = text.stem
                        text_path = str(text)  # Convert to string

                        # Find rows where name matches text_name
                        mask = df['name'] == text_name

                        # Add new columns if they don't exist
                        if 'label_name' not in df.columns:
                            df['label_name'] = None
                        if 'label_name_ext' not in df.columns:
                            df['label_name_ext'] = None
                        if 'label_path' not in df.columns:
                            df['label_path'] = None

                        # Update matching rows
                        df.loc[mask, 'label_name'] = text_name
                        df.loc[mask, 'label_name_ext'] = text_name_ext
                        df.loc[mask, 'label_path'] = text_path

                        print(f"Updated {mask.sum()} rows for {text_name}")

    # Save the updated DataFrame
    df.to_csv("output_with_labels.csv", index=False)


# Call the function
adding_labels()
