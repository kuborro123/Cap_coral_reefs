tag = "C:/Users/20231807/Documents/GitHub/Cap_coral_reefs/New_version_of_the_project/"

paths = ["train", "val", "test"]



for path in paths:
        # Read the original file
        with open(f"splits/{path}.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Write back with the tag removed
        with open(f"splits/{path}.txt", "w", encoding="utf-8") as file:
            for line in lines:
                cleaned_line = line.replace(tag, "").strip('"')  # remove tag + any stray quotes
                file.write(cleaned_line + "\n")
