import pandas as pd
import os
import glob

base = r"D:\FYP\code\archive"

# Read all 4 CSV files
mass_train = pd.read_csv(
    os.path.join(base, "csv", "mass_case_description_train_set.csv")
)
mass_test = pd.read_csv(os.path.join(base, "csv", "mass_case_description_test_set.csv"))
calc_train = pd.read_csv(
    os.path.join(base, "csv", "calc_case_description_train_set.csv")
)
calc_test = pd.read_csv(os.path.join(base, "csv", "calc_case_description_test_set.csv"))

print("=== Row Counts ===")
print(f"Mass Train: {len(mass_train)} rows")
print(f"Mass Test:  {len(mass_test)} rows")
print(f"Calc Train: {len(calc_train)} rows")
print(f"Calc Test:  {len(calc_test)} rows")
print(
    f"Total:      {len(mass_train) + len(mass_test) + len(calc_train) + len(calc_test)} rows"
)

print("\n=== Pathology Distribution ===")
for name, df in [
    ("Mass Train", mass_train),
    ("Mass Test", mass_test),
    ("Calc Train", calc_train),
    ("Calc Test", calc_test),
]:
    print(f"\n{name}:")
    print(df["pathology"].value_counts().to_string())

print("\n=== Combined Train Pathology ===")
combined_train = pd.concat([mass_train, calc_train])
print(combined_train["pathology"].value_counts().to_string())

print("\n=== Combined Test Pathology ===")
combined_test = pd.concat([mass_test, calc_test])
print(combined_test["pathology"].value_counts().to_string())

print("\n=== Binary (BENIGN_WITHOUT_CALLBACK -> BENIGN) ===")
for name, df in [("Train", combined_train), ("Test", combined_test)]:
    binary = df["pathology"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    print(f"{name}: {binary.value_counts().to_string()}")
    total = len(binary)
    benign = (binary == "BENIGN").sum()
    malignant = (binary == "MALIGNANT").sum()
    print(f"  Ratio B:M = {benign / total:.1%} : {malignant / total:.1%}")

print("\n=== Unique Patients ===")
print(f"Train patients: {combined_train['patient_id'].nunique()}")
print(f"Test patients: {combined_test['patient_id'].nunique()}")

# Check image path resolution
print("\n=== Image Path Resolution Check ===")
jpeg_dir = os.path.join(base, "jpeg")
all_jpeg_dirs = set(os.listdir(jpeg_dir))
print(f"Total directories in jpeg/: {len(all_jpeg_dirs)}")

# Try to resolve a few paths from mass_train
sample_paths = mass_train["cropped image file path"].dropna().head(5).tolist()
print(f"\nSample cropped image paths from CSV:")
for p in sample_paths:
    p_clean = str(p).strip().strip('"').strip()
    parts = p_clean.replace("\\", "/").split("/")
    print(f"  Path: {parts[0]}/.../{parts[-1]}")
    # Check if the folder name (first part) exists
    first_dir = parts[0] if len(parts) > 0 else ""
    # Try matching by looking for the DICOM UID parts
    if len(parts) >= 3:
        uid1 = parts[1] if len(parts) > 1 else ""
        uid2 = parts[2] if len(parts) > 2 else ""
        # Check if uid1 or uid2 is a directory name
        found_uid1 = uid1 in all_jpeg_dirs
        found_uid2 = uid2 in all_jpeg_dirs
        print(f"    UID1 ({uid1[:30]}...) found: {found_uid1}")
        print(f"    UID2 ({uid2[:30]}...) found: {found_uid2}")

# Check actual jpeg files
print("\n=== Sample JPEG directory contents ===")
sample_dirs = list(all_jpeg_dirs)[:3]
for d in sample_dirs:
    full_path = os.path.join(jpeg_dir, d)
    files = os.listdir(full_path)
    print(f"  {d[:40]}...: {files}")
