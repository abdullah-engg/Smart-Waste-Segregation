import os

def rename_audio_in_folder(folder_path, class_name):
    """
    Rename all WAV files in the specified folder to a uniform format:
    <class_name>_001.wav, <class_name>_002.wav, etc.
    """
    supported_extensions = ('.wav',)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]
    files.sort()  # Sort alphabetically

    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{class_name}_{i:03d}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        if os.path.exists(dst):
            print(f"⚠️ Skipped: {filename} (target {new_name} already exists)")
            continue

        os.rename(src, dst)
        print(f"Renamed: {filename} --> {new_name}")

    print(f"\n✅ Finished renaming audio in '{folder_path}' with prefix '{class_name}'")

# === 🔧 CHANGE THESE PATHS TO YOUR CLASS FOLDERS ===
class_folders = {
    "bottle": r"C:\D disc\proto1\dataset\bottle",
    "paper": r"C:\D disc\proto1\dataset\paper",
    "can": r"C:\D disc\proto1\dataset\can"
}

for class_name, folder_path in class_folders.items():
    rename_audio_in_folder(folder_path, class_name)
