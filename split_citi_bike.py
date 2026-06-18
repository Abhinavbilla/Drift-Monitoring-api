import pandas as pd
import csv
import os
import glob


def read_arff_file(file_path, chunk_size=50000):
    """
    Memory-efficient ARFF reader. Instead of building one giant
    list-of-lists for the whole file (which caused the MemoryError
    on large files like the Citi Bike dataset), this streams rows
    in chunks and builds the DataFrame incrementally.
    """
    column_names = []
    chunks = []
    current_chunk = []
    in_data = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith('@RELATION'):
                continue
            if line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                if len(parts) >= 3:
                    attr = parts[1].strip("'\"")
                    column_names.append(attr)
                continue
            if line.upper().startswith('@DATA'):
                in_data = True
                continue
            if in_data and line:
                reader = csv.reader([line], skipinitialspace=True)
                row = next(reader)
                current_chunk.append([x.strip() for x in row])

                # Flush this chunk into a DataFrame and reset, instead of
                # holding every row in memory simultaneously
                if len(current_chunk) >= chunk_size:
                    chunks.append(pd.DataFrame(current_chunk, columns=column_names))
                    current_chunk = []

    if not column_names:
        raise ValueError("No @ATTRIBUTE lines found – file may not be valid ARFF.")

    # Flush any remaining rows
    if current_chunk:
        chunks.append(pd.DataFrame(current_chunk, columns=column_names))

    df = pd.concat(chunks, ignore_index=True)

    # Coerce numeric columns column-by-column (pandas 3.0 compatible)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    return df


# ============================================
# Main – auto-find the ARFF file
# ============================================
# FIX: glob.glob("data/**/*.arff") matched dataset.arff first because
# glob does not guarantee any particular order and there are TWO .arff
# files in data/. Search specifically for "Citi" in the filename FIRST,
# and only fall back to a generic *.arff match if that fails.
arff_files = glob.glob("data/**/*Citi*.arff", recursive=True)
if not arff_files:
    arff_files = glob.glob("**/*Citi*.arff", recursive=True)
if not arff_files:
    arff_files = glob.glob("**/*Citi*", recursive=True)

if not arff_files:
    print("❌ Could not find an ARFF file containing 'Citi' anywhere in the project.")
    print("   Make sure 'New-York-Citi-Bike-Trip-Duration-2016.arff' exists under data/.")
    exit(1)

ARFF_FILE = arff_files[0]
print(f"Found ARFF file: {ARFF_FILE}")

print("Loading ARFF file (streaming in chunks to avoid MemoryError)...")
df = read_arff_file(ARFF_FILE)
print(f"Loaded {len(df)} rows.")
print(f"Columns found: {list(df.columns)}")

# Find the pickup datetime column
pickup_col = None
for col in df.columns:
    if 'pickup' in col.lower() and 'datetime' in col.lower():
        pickup_col = col
        break

if pickup_col is None:
    raise KeyError(
        "Could not find a 'pickup_datetime' column. Available columns: "
        + ", ".join(df.columns)
    )

df[pickup_col] = pd.to_datetime(df[pickup_col])
df['month'] = df[pickup_col].dt.month

baseline = df[df['month'] <= 3].copy()
production = df[df['month'] > 3].copy()

print(f"Baseline: {len(baseline)} rows")
print(f"Production: {len(production)} rows")

os.makedirs("tests", exist_ok=True)

baseline.to_csv("tests/citi_bike_baseline.csv", index=False)
production.to_csv("tests/citi_bike_production.csv", index=False)

print("✅ Files saved: tests/citi_bike_baseline.csv and tests/citi_bike_production.csv")