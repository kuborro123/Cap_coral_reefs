from google.cloud import storage
import os

bucket_name = "rs_storage_open"
base_prefix = "coral_bleaching/reef_support/UNAL_BLEACHING_TAYRONA/"
prefix_images = base_prefix + "images/"
prefix_bleached = base_prefix + "masks_bleached/"
prefix_non_bleached = base_prefix + "masks_non_bleached/"

project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
images_dir = os.path.join(data_dir, "images")
bleached_dir = os.path.join(data_dir, "masks_bleached")
non_bleached_dir = os.path.join(data_dir, "masks_non_bleached")

for d in [images_dir, bleached_dir, non_bleached_dir]:
    os.makedirs(d, exist_ok=True)

print("Connecting to Google Cloud Storage...")
client = storage.Client.create_anonymous_client()
bucket = client.bucket(bucket_name)

def list_files(prefix):
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png"))]

image_files = list_files(prefix_images)
bleached_files = list_files(prefix_bleached)
non_bleached_files = list_files(prefix_non_bleached)

print(f"Found: {len(image_files)} images, {len(bleached_files)} bleached masks, {len(non_bleached_files)} non-bleached masks")

def download_files(files, local_dir):
    for remote_path in files:
        fname = os.path.basename(remote_path)
        local_path = os.path.join(local_dir, fname)
        if not os.path.exists(local_path):
            bucket.blob(remote_path).download_to_filename(local_path)
            print(f"Downloaded {remote_path}")

download_files(image_files, images_dir)
download_files(bleached_files, bleached_dir)
download_files(non_bleached_files, non_bleached_dir)

print("All files downloaded")
