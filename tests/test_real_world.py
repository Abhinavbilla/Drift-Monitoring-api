import pandas as pd
import sys
import os
import time

# Tell Python to look in the parent directory for the utils folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from utils.profiler import profile_columns
except ModuleNotFoundError:
    # Secondary fallback lookup path strategy
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from utils.profiler import profile_columns

def run_dynamic_census_benchmark():
    print("=" * 75)
    print("🇳🇿 RUNNING DYNAMIC CENSUS AUTO-DISCOVERY & BENCHMARK")
    print("=" * 75)

    # Directories to completely bypass to save execution cycles
    ignore_dirs = {'.git', 'venv', '.venv', '__pycache__', 'env', '.pytest_cache'}
    
    csv_files = []
    
    # 1. Scan the immediate root workspace directory
    for item in os.listdir('.'):
        if item.endswith('.csv') and os.path.isfile(item):
            csv_files.append(os.path.abspath(item))
            
    # 2. Walk through any subdirectories just in case the files are tucked away
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignore_dirs] # filter ignored directories in-place
        for f in files:
            if f.endswith('.csv'):
                full_path = os.path.abspath(os.path.join(root, f))
                if full_path not in csv_files:
                    csv_files.append(full_path)

    # If no tracking targets found, print debug info
    if not csv_files:
        print("\n❌ Error: Could not find ANY CSV files anywhere in your project workspace.")
        print(f"Current working directory scanned: {os.getcwd()}")
        print("Please verify your census CSV files are located somewhere inside this directory.")
        return

    print(f"🔍 Auto-Discovery System found {len(csv_files)} target CSV files inside your workspace.\n")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        display_name = file_name.replace(".csv", "").replace("-", " ").upper()
        print(f"\n📥 Processing: {display_name}")
        print(f"   📂 Disk Location: {file_path}")
        
        try:
            start_time = time.time()
            # Set nrows=50000 for high-throughput scaling and RAM protection
            df = pd.read_csv(file_path, nrows=50000, low_memory=False)
            load_time = time.time() - start_time
            print(f"   ↳ Loaded {len(df):,} rows, {len(df.columns)} columns in {load_time:.2f}s.")

            # Execute your backend monitoring profiler logic
            profile_start = time.time()
            profiles = profile_columns(df)
            profile_time = time.time() - profile_start
            print(f"   ↳ Profiled in {profile_time:.4f}s.")

            print(f"   📊 {'COLUMN NAME':<40} | {'DECISION':<15} | {'REASON'}")
            print(f"   {'='*95}")
            
            for p in profiles:
                col = p["name"]
                decision = p["monitor"]
                reason = p["reason"]

                # Convert boolean/string decisions into clean monitoring layout indicators
                if decision is True: decision_str = "📈 Continuous"
                elif decision == "Categorical": decision_str = "🔠 Categorical"
                elif decision is False: decision_str = "❌ Ignore"
                else: decision_str = "⚠️ Review"

                print(f"   • {col:<40} | {decision_str:<15} | {reason}")
        except Exception as e:
            print(f"   💥 Error reading or profiling file: {e}")

if __name__ == "__main__":
    run_dynamic_census_benchmark()