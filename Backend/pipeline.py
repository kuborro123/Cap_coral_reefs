import pandas as pd
from PIL import Image
import os
import numpy as np


# Function to connect proper CSV
def con_css_index(name: str):
    csv = pd.read_csv(name)
    df = pd.DataFrame(csv)
    return df

main_csv = "reef_dataset_index_with_split.csv"
def adding_paths():
    df = con_css_index(main_csv)
    if df.empty:
        print("Could not load reef_dataset_index.csv. Exiting.")
        return

    folder_path_bleached = "masks_bleached"
    folder_path_non_bleached = "masks_non_bleached"

    # If "name" has extensions, split them
    base, ext = os.path.splitext(df["name"].iloc[0])
    if ext == "":
        # Assume all are PNG if no extension
        ext = ".png"

    # Build paths
    df["mask_bleached_path"] = df["name"].apply(
        lambda x: os.path.join(folder_path_bleached, f"{os.path.splitext(x)[0]}_bleached{ext}")
    )

    df["mask_non_bleached_path"] = df["name"].apply(
        lambda x: os.path.join(folder_path_non_bleached, f"{os.path.splitext(x)[0]}_non_bleached{ext}")
    )

    # Optionally check if the files exist
    df["mask_bleached_exists"] = df["mask_bleached_path"].apply(os.path.exists)
    df["mask_non_bleached_exists"] = df["mask_non_bleached_path"].apply(os.path.exists)

    # Save back to same CSV
    df.to_csv(main_csv, index=False)

    return df


def path_to_multimask():
    df = con_css_index(main_csv)
    if df.empty:
        print("Could not load reef_dataset_index.csv. Exiting.")
        return

    folder_path_multi = "masks_multi"

    # Extract extension from first row (assume consistent)
    base, ext = os.path.splitext(df["name"].iloc[0])
    if ext == "":
        ext = ".png"

    # Build the multi-mask path column
    df["mask_multi_path"] = df["name"].apply(
        lambda x: os.path.join(folder_path_multi, f"{os.path.splitext(x)[0].lower()}_multi{ext}")
    )

    # Save back to same CSV
    df.to_csv(main_csv, index=False)

    return df

def creating_site_transpect():

    df = con_css_index(main_csv)
    if df.empty:
        print("Could not load reef_dataset_index.csv. Exiting.")
        return

    # Split "name" into multiple columns
    parts = df["name"].str.split("_", expand=True)

    # Assign the parts to new columns
    df["colony"] = parts[0]
    df["site"] = parts[1]
    df["time"] = parts[2]
    df["transect"] = parts[3]
    df["date"] = parts[4]
    df["diver"] = parts[5]

    df["date"] = pd.to_datetime(parts[4], format="%d%b%y", errors="coerce").dt.strftime("%d.%m.%Y")

    # Save back to same CSV
    df.to_csv(main_csv, index=False)



def get_num_pixels(filepath):
    with Image.open(filepath) as im:
        width, height = im.size
    return width * height

def columns_pixel():
    df = con_css_index(main_csv)
    if df.empty:
        print("Could not load reef_dataset_index.csv. Exiting.")
        return
    df["px_size_image"] = df["image_path"].apply(get_num_pixels)
    df["px_size_multimask"] = df["mask_multi_path"].apply(get_num_pixels)

    df.to_csv(main_csv, index=False)

def get_multimask_pixel():
    """
    For each mask in the CSV, calculate pixel statistics.
    Adds new columns to the DataFrame.
    """
    # Load your CSV
    df = pd.read_csv(main_csv)

    # Create empty columns
    df["pix_background"] = 0
    df["pix_healthy"] = 0
    df["pix_bleached"] = 0

    # Loop through rows
    for idx, row in df.iterrows():
        mask_path = row["mask_multi_path"]
        mask = np.array(Image.open(mask_path))

        df.at[idx, "pix_background"] = int(np.sum(mask == 0))
        df.at[idx, "pix_healthy"] = int(np.sum(mask == 1))
        df.at[idx, "pix_bleached"] = int(np.sum(mask == 2))

        df.to_csv(main_csv, index=False)

    return None

def computing_pixels():
    df = pd.read_csv(main_csv)

    df["coral_pix"] = df["pix_healthy"] + df["pix_bleached"]
    df["coral_cover_%"] = (df["coral_pix"] / df["px_size_multimask"]) * 100

    df["bleaching_%"] = np.where(
        df["coral_pix"] > 0,
        (df["pix_bleached"] / df["coral_pix"]) * 100,
        0.0
    )

    df.to_csv(main_csv, index=False)
    return

def renaming_columns():
    df = con_css_index(main_csv)
    df.rename(columns={'colony': 'site_id', 'site': 'zone'}, inplace=True)
    df.to_csv(main_csv, index=False)
    return None

renaming_columns()

