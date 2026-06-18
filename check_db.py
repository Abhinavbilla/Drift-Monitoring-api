import sqlite3
import json

PROJECT_ID = "citi_bike_v1"  # Your model ID

print("=" * 60)
print(f"Checking categorical baselines for project: {PROJECT_ID}")
print("=" * 60)

conn = sqlite3.connect("drift.db")
cursor = conn.cursor()

# 1. Check the baselines table structure
cursor.execute("PRAGMA table_info(baselines)")
columns = cursor.fetchall()
print("\n✅ Baselines table columns:")
for col in columns:
    print(f"  - {col[1]} ({col[2]})")

# 2. Check if the project exists in baselines
cursor.execute(
    "SELECT project_id, feature_types, reference_data, iqr_fences, categorical_baselines FROM baselines WHERE project_id = ?",
    (PROJECT_ID,)
)
row = cursor.fetchone()

if not row:
    print(f"\n❌ Project '{PROJECT_ID}' not found in baselines table.")
    conn.close()
    exit(1)

print(f"\n✅ Project '{PROJECT_ID}' found in baselines table.")

project_id, feature_types, reference_data, iqr_fences, categorical_baselines = row

# 3. Check categorical_baselines
print("\n📊 categorical_baselines column:")
if categorical_baselines:
    cat_data = json.loads(categorical_baselines)
    print(f"  - Contains {len(cat_data)} features: {list(cat_data.keys())}")
    for feature, baseline in cat_data.items():
        print(f"\n  🔹 {feature}:")
        # Show first 5 categories
        if isinstance(baseline, dict):
            items = list(baseline.items())
            print(f"     Total categories: {len(items)}")
            print(f"     Sample: {items[:5]}")
            # Check if month has only 1,2,3
            if feature == "month":
                keys = set(baseline.keys())
                print(f"     Unique month values in baseline: {sorted(keys)}")
        else:
            print(f"     Type: {type(baseline)} (expected dict)")
else:
    print("  ❌ categorical_baselines is EMPTY or NULL")
    print("     → This explains why categorical drift (gender_id, month) is not being detected!")

# 4. Also check feature_types to see how month is stored
print("\n📊 feature_types:")
if feature_types:
    ft = json.loads(feature_types)
    print(f"  {ft}")
    if "month" in ft:
        print(f"  month type: {ft['month']}")
    if "gender_id" in ft:
        print(f"  gender_id type: {ft['gender_id']}")

# 5. Check if month is in the reference_data (for the PSI calculation)
print("\n📊 reference_data (first feature sample):")
if reference_data:
    ref = json.loads(reference_data)
    print(f"  Total features in reference_data: {len(ref)}")
    if "month" in ref:
        sample = ref["month"][:10]
        print(f"  month sample: {sample}")
    if "gender_id" in ref:
        sample = ref["gender_id"][:10]
        print(f"  gender_id sample: {sample}")

conn.close()

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
if not categorical_baselines:
    print("❌ categorical_baselines is empty → categorical drift detection is disabled.")
    print("   Fix: Ensure insert_baseline() in crud.py is storing the categorical frequency tables.")
elif categorical_baselines and "month" not in cat_data:
    print("❌ 'month' is not in categorical_baselines → your API is not comparing it.")
    print("   Fix: Check that month was correctly classified as categorical during fit.")
elif categorical_baselines and "month" in cat_data:
    print("✅ 'month' IS stored in categorical_baselines.")
    print("   → The issue is in the comparison logic inside _check_categorical_drift().")
    print("   Check for: type mismatches (str vs int), missing categories, or epsilon handling.")