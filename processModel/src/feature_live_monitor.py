# src_process/feature_live_monitor.py
"""
System-wide memory forensics monitor.
Collects aggregate statistics matching CICMalMem training features.
Unlike per-process monitoring, this analyzes the ENTIRE SYSTEM state.
"""
import joblib, os, time, numpy as np, pandas as pd
import psutil
from collections import Counter, defaultdict

MODEL_PATH = "../models/rf_process.pkl"
SCALER_PATH = "../models/scaler.pkl"
FEATURES_PATH = "../models/feature_columns.txt"


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found in models/. Run training first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = []
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'r') as f:
            feature_cols = [l.strip() for l in f if l.strip()]
    return model, scaler, feature_cols


def collect_system_wide_features():
    """
    Collect system-wide statistics that approximate Volatility plugin outputs.
    This returns ONE feature vector representing the ENTIRE SYSTEM state.
    """
    features = {}

    # Get all accessible processes
    all_procs = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            p = psutil.Process(proc.info['pid'])
            all_procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # ============= PSLIST FEATURES =============
    # Total number of processes
    features['pslist.nproc'] = len(all_procs)

    # Average threads per process
    thread_counts = []
    for p in all_procs:
        try:
            thread_counts.append(p.num_threads())
        except:
            pass
    features['pslist.avg_threads'] = np.mean(thread_counts) if thread_counts else 0

    # Number of 64-bit processes (approximation - count processes with higher memory usage)
    # Note: This is an approximation; actual detection requires platform-specific APIs
    features['pslist.nprocs64bit'] = len(all_procs)  # Most modern systems run 64-bit

    # Average handles per process
    handle_counts = []
    for p in all_procs:
        try:
            # On Windows: num_handles(), on Unix: approximate with open files + connections
            if hasattr(p, 'num_handles'):
                handle_counts.append(p.num_handles())
            else:
                # Unix approximation: files + connections
                try:
                    h_count = len(p.open_files()) + len(p.connections())
                    handle_counts.append(h_count)
                except:
                    pass
        except:
            pass
    features['pslist.avg_handlers'] = np.mean(handle_counts) if handle_counts else 0

    # ============= DLLLIST FEATURES =============
    # Approximate DLL/library loading statistics
    total_libs = 0
    proc_with_libs = 0
    for p in all_procs:
        try:
            # memory_maps() returns loaded libraries/modules
            maps = p.memory_maps()
            if maps:
                total_libs += len(maps)
                proc_with_libs += 1
        except:
            pass

    features['dlllist.ndlls'] = total_libs
    features['dlllist.avg_dlls_per_proc'] = total_libs / len(all_procs) if all_procs else 0

    # ============= HANDLES FEATURES =============
    total_handles = sum(handle_counts) if handle_counts else 0
    features['handles.nhandles'] = total_handles
    features['handles.avg_handles_per_proc'] = np.mean(handle_counts) if handle_counts else 0

    # Handle type breakdown (approximated)
    # Volatility categorizes handles by type; we approximate with connections and files
    total_connections = 0
    total_files = 0
    total_threads = 0

    for p in all_procs:
        try:
            total_connections += len(p.connections())
        except:
            pass
        try:
            total_files += len(p.open_files())
        except:
            pass
        try:
            total_threads += p.num_threads()
        except:
            pass

    # Approximate handle type distributions
    features['handles.nport'] = total_connections  # Network ports/sockets
    features['handles.nevent'] = int(total_handles * 0.1)  # Events (approximation)
    features['handles.ndesktop'] = len(all_procs)  # Roughly one per process
    features['handles.nkey'] = int(total_handles * 0.05)  # Registry keys (approximation)
    features['handles.nthread'] = total_threads
    features['handles.ndirectory'] = int(total_handles * 0.02)  # Directory handles
    features['handles.nsemaphore'] = int(total_handles * 0.03)  # Semaphores
    features['handles.nsection'] = int(total_handles * 0.08)  # Memory sections
    features['handles.nmutant'] = int(total_handles * 0.02)  # Mutexes

    # ============= LDRMODULES FEATURES =============
    # Module loading anomalies (processes with unusual library patterns)
    # These detect rootkit-style hiding; we approximate with outlier detection
    lib_counts = []
    for p in all_procs:
        try:
            lib_counts.append(len(p.memory_maps()))
        except:
            lib_counts.append(0)

    if lib_counts:
        median_libs = np.median(lib_counts)
        # Count processes with suspiciously few libraries (potential hiding)
        outliers_low = sum(1 for x in lib_counts if x < median_libs * 0.3)
        outliers_high = sum(1 for x in lib_counts if x > median_libs * 3)

        features['ldrmodules.not_in_load'] = outliers_low
        features['ldrmodules.not_in_init'] = outliers_low
        features['ldrmodules.not_in_mem'] = outliers_low
        features['ldrmodules.not_in_load_avg'] = outliers_low / len(all_procs)
        features['ldrmodules.not_in_init_avg'] = outliers_low / len(all_procs)
        features['ldrmodules.not_in_mem_avg'] = outliers_low / len(all_procs)
    else:
        for key in ['not_in_load', 'not_in_init', 'not_in_mem',
                    'not_in_load_avg', 'not_in_init_avg', 'not_in_mem_avg']:
            features[f'ldrmodules.{key}'] = 0

    # ============= MALFIND FEATURES =============
    # Code injection detection (approximated by unusual memory patterns)
    suspicious_memory = 0
    unique_suspicious = 0
    total_commit = 0

    for p in all_procs:
        try:
            mem_info = p.memory_info()
            total_commit += mem_info.rss

            # Flag processes with unusually high private memory (possible injection)
            if mem_info.rss > 500 * 1024 * 1024:  # > 500MB
                suspicious_memory += 1

            # Check for unusual memory patterns via maps
            try:
                maps = p.memory_maps()
                # Suspiciously many anonymous mappings might indicate injection
                anon_maps = sum(1 for m in maps if '[anon' in m.path.lower() or 'heap' in m.path.lower())
                if anon_maps > 50:
                    unique_suspicious += 1
            except:
                pass
        except:
            pass

    features['malfind.ninjections'] = suspicious_memory
    features['malfind.commitCharge'] = total_commit / (1024 * 1024)  # MB
    features['malfind.protection'] = suspicious_memory  # Processes with suspicious protections
    features['malfind.uniqueInjections'] = unique_suspicious

    # ============= PSXVIEW FEATURES =============
    # Process hiding detection (cross-view inconsistencies)
    # In live system, we can't easily detect these without kernel access
    # Set to 0 as baseline (no hiding detected in normal operation)
    psxview_keys = [
        'not_in_pslist', 'not_in_eprocess_pool', 'not_in_ethread_pool',
        'not_in_csrss_handles', 'not_in_session', 'not_in_deskthrd',
        'not_in_pslist_false_avg', 'not_in_eprocess_pool_false_avg',
        'not_in_ethread_pool_false_avg', 'not_in_csrss_handles_false_avg',
        'not_in_session_false_avg', 'not_in_deskthrd_false_avg'
    ]
    for key in psxview_keys:
        features[f'psxview.{key}'] = 0  # Baseline: no hiding detected

    # ============= MODULES FEATURES =============
    # Kernel modules (drivers) - approximate with total unique libraries
    unique_modules = set()
    for p in all_procs:
        try:
            for m in p.memory_maps():
                unique_modules.add(m.path)
        except:
            pass
    features['modules.nmodules'] = len(unique_modules)

    # ============= SVCSCAN FEATURES =============
    # Windows services - approximate with daemon/background processes
    # On Unix: count processes running as root or system services
    service_like = 0
    kernel_drivers = 0

    for p in all_procs:
        try:
            # Service-like: low CPU, long-running, specific names
            name = p.name().lower()
            if any(x in name for x in ['service', 'daemon', 'd', 'server', 'agent']):
                service_like += 1

            # Kernel driver approximation: system processes
            if p.username() in ['root', 'SYSTEM', '_windowserver', 'NT AUTHORITY\\SYSTEM']:
                kernel_drivers += 1
        except:
            pass

    features['svcscan.nservices'] = service_like
    features['svcscan.kernel_drivers'] = kernel_drivers
    features['svcscan.fs_drivers'] = int(kernel_drivers * 0.3)  # File system drivers subset
    features['svcscan.process_services'] = service_like
    features['svcscan.shared_process_services'] = int(service_like * 0.5)
    features['svcscan.interactive_process_services'] = int(service_like * 0.2)
    features['svcscan.nactive'] = service_like

    # ============= CALLBACKS FEATURES =============
    # Kernel callbacks - not accessible from userland without special tools
    # Set baseline values
    features['callbacks.ncallbacks'] = 0
    features['callbacks.nanonymous'] = 0
    features['callbacks.ngeneric'] = 0

    return features


def build_feature_vector(feature_cols, system_features):
    """Build feature vector in exact order expected by model"""
    row = [float(system_features.get(c, 0.0)) for c in feature_cols]
    return np.array(row, dtype=float).reshape(1, -1)


def main(poll_interval=5.0, threshold=0.75):
    """
    Main monitoring loop - analyzes SYSTEM-WIDE state periodically.

    Args:
        poll_interval: Seconds between scans
        threshold: Malicious probability threshold (0.5-0.9, default 0.75)
    """
    print("[monitor] Loading model artifacts...")
    model, scaler, feature_cols = load_artifacts()

    if not feature_cols:
        print("[monitor] ERROR: feature_columns.txt is empty!")
        return

    print(f"[monitor] Model expects {len(feature_cols)} features")
    print(f"[monitor] Features: {feature_cols[:5]}...")

    # Test feature collection
    print("\n[monitor] Testing feature collection...")
    test_features = collect_system_wide_features()

    matched = sum(1 for c in feature_cols if c in test_features)
    coverage = matched / len(feature_cols)

    print(f"[monitor] Collected {len(test_features)} system features")
    print(f"[monitor] Coverage: {coverage * 100:.1f}% ({matched}/{len(feature_cols)} features)")

    if coverage < 0.8:
        print("[monitor] ⚠️  WARNING: Low feature coverage!")
        print("[monitor] Some features are approximations or unavailable without kernel access.")
        print("[monitor] Predictions may be less accurate than training environment.")

    print(f"\n[monitor] Starting system monitoring (poll every {poll_interval}s)")
    print("[monitor] This monitors ENTIRE SYSTEM state, not individual processes")
    print("[monitor] Press Ctrl-C to stop.\n")

    scan_count = 0

    while True:
        scan_count += 1
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"{'=' * 80}")
        print(f"SYSTEM SCAN #{scan_count} at {timestamp}")
        print('=' * 80)

        try:
            # Collect system-wide features
            sys_features = collect_system_wide_features()

            # Build feature vector
            feature_vec = build_feature_vector(feature_cols, sys_features)
            feature_vec_scaled = scaler.transform(feature_vec)

            # Make prediction with adjusted threshold
            try:
                proba = model.predict_proba(feature_vec_scaled)[0]
                benign_prob = proba[0]
                malicious_prob = proba[1]

                # Use higher threshold to reduce false positives
                # Training data may have different system characteristics
                MALICIOUS_THRESHOLD = threshold
                prediction = 1 if malicious_prob >= MALICIOUS_THRESHOLD else 0

            except:
                # Fallback to default prediction if probabilities unavailable
                prediction = model.predict(feature_vec_scaled)[0]
                benign_prob = None
                malicious_prob = None

            # Display results
            status = "🚨 MALICIOUS ACTIVITY DETECTED" if prediction == 1 else "✓ SYSTEM APPEARS CLEAN"
            color = "RED" if prediction == 1 else "GREEN"

            print(f"\n{status}")
            if malicious_prob is not None:
                print(f"Confidence: Benign={benign_prob:.1%}, Malicious={malicious_prob:.1%}")
                print(f"Threshold: {threshold:.1%} (scores >= {threshold:.1%} flagged as malicious)")

                # Show risk level based on proximity to threshold
                if malicious_prob >= threshold:
                    risk = "HIGH"
                elif malicious_prob >= threshold - 0.1:
                    risk = "MODERATE (near threshold)"
                else:
                    risk = "LOW"
                print(f"Risk Level: {risk}")

            # Show key metrics
            print(f"\nKey System Metrics:")
            print(f"  Total Processes: {sys_features.get('pslist.nproc', 0)}")
            print(f"  Avg Threads/Proc: {sys_features.get('pslist.avg_threads', 0):.1f}")
            print(f"  Total Handles: {sys_features.get('handles.nhandles', 0)}")
            print(f"  Total Connections: {sys_features.get('handles.nport', 0)}")
            print(f"  Suspicious Memory: {sys_features.get('malfind.ninjections', 0)} processes")
            print(f"  Memory Commit: {sys_features.get('malfind.commitCharge', 0):.0f} MB")
            print(f"  Service Processes: {sys_features.get('svcscan.nservices', 0)}")

            if prediction == 1:
                print(f"\n⚠️  RECOMMENDED ACTIONS:")
                print(f"  1. Review processes with high memory usage")
                print(f"  2. Check for unusual network connections")
                print(f"  3. Scan with dedicated antivirus/EDR tools")
                print(f"  4. Review system logs for anomalies")

        except Exception as e:
            print(f"❌ Error during scan: {e}")
            import traceback
            traceback.print_exc()

        print(f"\nNext scan in {poll_interval}s...\n")
        time.sleep(poll_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='System-wide malware detection monitor (matches CICMalMem training)'
    )
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Scan interval in seconds (default: 5.0)')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Malicious probability threshold 0.0-1.0 (default: 0.75, higher=fewer alerts)')
    args = parser.parse_args()

    try:
        main(poll_interval=args.interval, threshold=args.threshold)
    except KeyboardInterrupt:
        print("\n[monitor] Stopped by user")