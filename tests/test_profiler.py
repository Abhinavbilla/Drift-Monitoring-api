import pandas as pd
import sys
import os

# Tell Python to look in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.profiler import profile_columns


def run_profiler_benchmark():
    print("🚀 Starting Profiler Benchmark Test...")

    # 1. Create a tricky, structured mock dataset
    data = {
        "User_ID": ["U101", "U102", "U103", "U104", "U105", "U106", "U107", "U108"], # Should be Ignore
        "Age": [25, 30, 45, 22, 50, 35, 28, 40],                                     # Should be Continuous
        "Salary": [50000.5, 60000.0, 120000.75, 45000.0, 90000.0, 75000.0, None, 80000.0], # Should be Continuous
        "Subscription_Tier": ["Basic", "Pro", "Basic", "Enterprise", "Pro", "Basic", "Basic", "Pro"], # Should be Categorical
        "Is_Active": [1, 0, 1, 1, 0, 1, 0, 1],                                       # Should be Categorical (Integer)
        "Signup_Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08"], # Should be Ignore (Datetime/Monotonic)
        "Blank_Column": [None, None, None, None, None, None, None, None]             # Should be Ignore (Empty)
    }
    df = pd.DataFrame(data)

    # 2. Define the exact "Ground Truth" we expect from our engine
    ground_truth = {
        "User_ID": False,             # False means "Ignore"
        "Age": True,                  # True means "Continuous"
        "Salary": True,               
        "Subscription_Tier": "Categorical", 
        "Is_Active": "Categorical",   
        "Signup_Date": False,         
        "Blank_Column": False         
    }

    # 3. Run the engine
    profiles = profile_columns(df)

    # 4. Calculate the Percentage Score
    correct_predictions = 0
    total_columns = len(ground_truth)

    print("\n📊 --- BENCHMARK RESULTS ---")
    for p in profiles:
        col_name = p["name"]
        predicted = p["monitor"]
        expected = ground_truth[col_name]
        
        # Check if the prediction matches the ground truth
        if predicted == expected:
            correct_predictions += 1
            print(f"✅ {col_name}: Correct ({predicted})")
        else:
            print(f"❌ {col_name}: Failed! Predicted '{predicted}', but expected '{expected}'")

    # Calculate final percentage
    accuracy = (correct_predictions / total_columns) * 100
    
    print("\n======================================")
    print(f"🎯 PROFILER ACCURACY SCORE: {accuracy:.2f}%")
    print("======================================")

if __name__ == "__main__":
    run_profiler_benchmark()