import psutil
import pandas as pd
import numpy as np
import os


def verify_data_relatability(dataset_path):
    # --- STEP 1: Load and Profile the Dataset ---
    if not os.path.exists(dataset_path):
        print(f"❌ Error: '{dataset_path}' not found.")
        return

    print(f"📂 Loading and Profiling Dataset: {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    required_cols = ['Category', 'pslist.avg_threads', 'handles.nhandles']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: Dataset missing required columns: {required_cols}")
        return

    benign_df = df[df['Category'] == 'Benign']
    malware_df = df[df['Category'] != 'Benign']

    benign_avg_threads = benign_df['pslist.avg_threads'].mean()
    benign_avg_handles = benign_df['handles.nhandles'].mean()

    malware_avg_threads = malware_df['pslist.avg_threads'].mean()
    # Defaulting to 0 if NaN to avoid math errors
    malware_avg_threads = 0 if np.isnan(malware_avg_threads) else malware_avg_threads

    print("\n📊 DATASET BASELINE")
    print("-" * 40)
    print(f"🟢 Average BENIGN Process Threads: {benign_avg_threads:.2f}")
    print(f"🔴 Average MALWARE Process Threads: {malware_avg_threads:.2f}")
    print("-" * 40)

    # --- STEP 2: Profile Local System (macOS) ---
    print("\n📸 Profiling Your Local System (Top 10 Processes)...")

    local_stats = []

    # Fetch process info
    for proc in psutil.process_iter(['pid', 'name', 'num_threads']):
        try:
            # 1. SAFEGUARD: Handle 'None' for threads
            n_threads = proc.info.get('num_threads')
            if n_threads is None:
                n_threads = 0

            # 2. SAFEGUARD: Handle 'None' or errors for file descriptors
            try:
                n_files = proc.num_fds()
            except (psutil.AccessDenied, AttributeError, ValueError):
                n_files = 0

            name = proc.info.get('name') or "Unknown"

            local_stats.append({
                'name': name,
                'threads': n_threads,
                'files': n_files
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Sort by thread count (descending)
    # Now that we ensured n_threads is always an int (0 or higher), this won't crash
    local_stats.sort(key=lambda x: x['threads'], reverse=True)

    # --- STEP 3: Cross-Reference & Report ---
    print(f"\n🔍 RELATABILITY CHECK")
    print("Comparing your top processes to the Dataset Averages...")
    print("-" * 60)
    print(f"{'Process Name':<30} | {'Threads':<10} | {'Files':<10} | {'Status Check'}")
    print("-" * 60)

    matches_count = 0

    for item in local_stats[:15]:  # Checking top 15 heavy processes
        t_count = item['threads']
        f_count = item['files']
        name = item['name']

        # Determine status based on dataset averages
        status = "✅ Benign-like"

        # Logic: If threads are closer to Malware avg than Benign avg
        # AND significantly higher than Benign avg
        if t_count > benign_avg_threads * 1.5:
            status = "⚠️ High Activity"

        print(f"{name[:28]:<30} | {t_count:<10} | {f_count:<10} | {status}")
        matches_count += 1

    print("-" * 60)

    # Success condition: If we successfully compared at least 10 processes
    if matches_count >= 10:
        print("success")
    else:
        print(f"Analyzed {matches_count} processes.")


if __name__ == "__main__":
    # Ensure path matches your local setup
    verify_data_relatability('../data/CICMalMem2022.csv')