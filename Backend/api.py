from google.cloud import storage
import os

# Anonymous client since rs_storage_open is public
client = storage.Client.create_anonymous_client()
bucket = client.bucket("rs_storage_open")

# Folder containing the photos
prefix = "coral_bleaching/reef_support/UNAL_BLEACHING_TAYRONA/images/"

# List all image files in that folder
print(f"Listing images in: {prefix}")
image_files = []
for blob in bucket.list_blobs(prefix=prefix):
    if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(blob.name)
        print(blob.name)

print(f"\nTotal images found: {len(image_files)}")

# Example: download the first image
if image_files:
    blob = bucket.blob(image_files[0])
    local_filename = os.path.basename(image_files[0])  # keep original name
    blob.download_to_filename(local_filename)
    print(f"Downloaded {image_files[0]} → {local_filename}")
