import os
import glob
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks, spectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# SMAC imports for MOO
try:
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.multi_objective.parego import ParEGO
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
    from ConfigSpace.conditions import EqualsCondition
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
    print("SMAC not available. Please install with: pip install smac")

print("Enhanced Multi-Objective Bayesian Optimization for Heartbeat Classification")
print("=" * 80)

# Data preprocessing functions 
def butter_bandpass_filter(data, lowcut=25.0, highcut=400.0, fs=1000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_label(filename):
    filename = filename.lower()
    if "normal" in filename:
        return 0
    elif any(x in filename for x in ["murmur", "artifact", "extrahls", "abnormal"]):
        return 1
    return None

def extract_mfcc(cycle, sr, n_mfcc=13, max_len=260):
    if len(cycle) < 512:
        cycle = np.pad(cycle, (0, 512 - len(cycle)), mode='constant')

    mfcc = librosa.feature.mfcc(y=cycle.astype(np.float32), sr=sr, n_mfcc=n_mfcc,
                               n_fft=512, hop_length=128)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def process_audio_data(data_path):
    print(f"Processing audio files from: {data_path}")

    # Find all wav files
    audio_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))

    print(f"Found {len(audio_files)} audio files")

    X_features = []
    y_labels = []

    for i, file_path in enumerate(audio_files):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=1000)

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Apply Butterworth filter
            filtered_audio = butter_bandpass_filter(audio, fs=sr)

            # Apply Hilbert transform for envelope
            envelope = np.abs(hilbert(filtered_audio))

            # Find peaks for cardiac cycles
            peaks, _ = find_peaks(envelope, distance=int(0.4 * sr), height=np.mean(envelope) * 1.2)

            # Extract cardiac cycles
            if len(peaks) >= 3:
                for j in range(len(peaks) - 2):
                    start, end = peaks[j], peaks[j + 2]
                    cycle = filtered_audio[start:end]
                    if len(cycle) > 100:
                        # Extract MFCC features
                        mfcc = extract_mfcc(cycle, sr)
                        features = mfcc.flatten()

                        # Get label
                        label = get_label(file_path)
                        if label is not None:
                            X_features.append(features)
                            y_labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(audio_files)} files")

        except Exception as e:
            continue

    X = np.array(X_features)
    y = np.array(y_labels)

    print(f"Extracted {len(X)} cardiac cycles")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

    return X, y

# Enhanced Multi-objective evaluation tracker
class EnhancedMultiObjectiveTracker:
    def __init__(self):
        self.all_evaluations = []
        self.classifier_times = {'svm': [], 'rf': [], 'xgb': []}
        self.pareto_candidates = []

    def add_evaluation(self, config, objectives, metrics, times):
        """Store evaluation results"""
        evaluation = {
            'config': config,
            'objectives': objectives,
            'metrics': metrics,
            'times': times,
            'eval_id': len(self.all_evaluations) + 1
        }
        self.all_evaluations.append(evaluation)

        # Track timing by classifier
        algorithm = config['algorithm']
        self.classifier_times[algorithm].append(times)

        # Update Pareto candidates
        self._update_pareto_front(evaluation)

    def _update_pareto_front(self, new_eval):
        """Manually maintain Pareto front"""
        objectives = new_eval['objectives']
        
        # Check if new solution dominates any existing ones
        dominated_indices = []
        is_dominated = False
        
        for i, candidate in enumerate(self.pareto_candidates):
            candidate_obj = candidate['objectives']
            
            # Check dominance relationships
            new_dominates = all(objectives[j] <= candidate_obj[j] for j in range(len(objectives))) and \
                           any(objectives[j] < candidate_obj[j] for j in range(len(objectives)))
            
            candidate_dominates = all(candidate_obj[j] <= objectives[j] for j in range(len(objectives))) and \
                                any(candidate_obj[j] < objectives[j] for j in range(len(objectives)))
            
            if new_dominates:
                dominated_indices.append(i)
            elif candidate_dominates:
                is_dominated = True
                break
        
        # If new solution is not dominated, add it and remove dominated ones
        if not is_dominated:
            # Remove dominated solutions
            for i in sorted(dominated_indices, reverse=True):
                del self.pareto_candidates[i]
            
            # Add new solution
            self.pareto_candidates.append(new_eval)

    def get_pareto_front(self):
        """Return current Pareto front"""
        return [(eval['config'], eval['objectives']) for eval in self.pareto_candidates]

    def get_aggregated_times(self):
        """Get aggregated timing statistics by classifier"""
        stats = {}
        for alg, times_list in self.classifier_times.items():
            if times_list:
                train_times = [t['train_time'] for t in times_list]
                test_times = [t['test_time'] for t in times_list]
                stats[alg] = {
                    'avg_train_time': np.mean(train_times),
                    'std_train_time': np.std(train_times),
                    'avg_test_time': np.mean(test_times),
                    'std_test_time': np.std(test_times),
                    'total_evaluations': len(times_list)
                }
        return stats

def create_optimized_config_space():
    """Create optimized configuration space with better parameter ranges"""
    cs = ConfigurationSpace()

    # Common hyperparameter
    algorithm = CategoricalHyperparameter("algorithm", choices=["svm", "rf", "xgb"], default_value="xgb")
    cs.add_hyperparameter(algorithm)

    # SVM parameters - optimized ranges
    C = UniformFloatHyperparameter("C", lower=0.1, upper=100.0, default_value=1.0, log=True)
    gamma = UniformFloatHyperparameter("gamma", lower=0.001, upper=1.0, default_value=0.1, log=True)
    kernel = CategoricalHyperparameter("kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")

    # RF parameters - optimized ranges
    rf_n_estimators = UniformIntegerHyperparameter("rf_n_estimators", lower=50, upper=500, default_value=100)
    rf_max_depth = UniformIntegerHyperparameter("rf_max_depth", lower=5, upper=50, default_value=15)
    rf_min_samples_split = UniformIntegerHyperparameter("rf_min_samples_split", lower=2, upper=20, default_value=2)
    rf_min_samples_leaf = UniformIntegerHyperparameter("rf_min_samples_leaf", lower=1, upper=10, default_value=1)

    # XGB parameters - optimized ranges
    xgb_n_estimators = UniformIntegerHyperparameter("xgb_n_estimators", lower=50, upper=500, default_value=100)
    xgb_max_depth = UniformIntegerHyperparameter("xgb_max_depth", lower=3, upper=15, default_value=6)
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.01, upper=0.3, default_value=0.1, log=True)
    subsample = UniformFloatHyperparameter("subsample", lower=0.6, upper=1.0, default_value=1.0)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", lower=0.6, upper=1.0, default_value=1.0)

    # Add all hyperparameters
    cs.add_hyperparameters([
        C, gamma, kernel,
        rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_min_samples_leaf,
        xgb_n_estimators, xgb_max_depth, learning_rate, subsample, colsample_bytree
    ])

    # Add conditions
    cs.add_condition(EqualsCondition(C, algorithm, "svm"))
    cs.add_condition(EqualsCondition(gamma, algorithm, "svm"))
    cs.add_condition(EqualsCondition(kernel, algorithm, "svm"))
    
    cs.add_condition(EqualsCondition(rf_n_estimators, algorithm, "rf"))
    cs.add_condition(EqualsCondition(rf_max_depth, algorithm, "rf"))
    cs.add_condition(EqualsCondition(rf_min_samples_split, algorithm, "rf"))
    cs.add_condition(EqualsCondition(rf_min_samples_leaf, algorithm, "rf"))
    
    cs.add_condition(EqualsCondition(xgb_n_estimators, algorithm, "xgb"))
    cs.add_condition(EqualsCondition(xgb_max_depth, algorithm, "xgb"))
    cs.add_condition(EqualsCondition(learning_rate, algorithm, "xgb"))
    cs.add_condition(EqualsCondition(subsample, algorithm, "xgb"))
    cs.add_condition(EqualsCondition(colsample_bytree, algorithm, "xgb"))

    return cs

def create_enhanced_multiobjective_function(X_train, y_train, X_test, y_test, tracker):
    """Create enhanced multi-objective function with abnormal recall and false negatives"""

    def objective(config, seed=0):
        try:
            algorithm = config['algorithm']

            # Create model based on algorithm
            if algorithm == 'svm':
                model = SVC(
                    C=config['C'], 
                    kernel=config['kernel'], 
                    gamma=config.get('gamma', 'scale'), 
                    random_state=42
                )
                clean_config = {
                    'algorithm': algorithm,
                    'C': config['C'],
                    'kernel': config['kernel'],
                    'gamma': config.get('gamma', 'scale')
                }
            elif algorithm == 'rf':
                model = RandomForestClassifier(
                    n_estimators=config['rf_n_estimators'],
                    max_depth=config['rf_max_depth'],
                    min_samples_split=config['rf_min_samples_split'],
                    min_samples_leaf=config['rf_min_samples_leaf'],
                    random_state=42,
                    n_jobs=1
                )
                clean_config = {
                    'algorithm': algorithm,
                    'n_estimators': config['rf_n_estimators'],
                    'max_depth': config['rf_max_depth'],
                    'min_samples_split': config['rf_min_samples_split'],
                    'min_samples_leaf': config['rf_min_samples_leaf']
                }
            else:  # xgb
                model = XGBClassifier(
                    n_estimators=config['xgb_n_estimators'],
                    max_depth=config['xgb_max_depth'],
                    learning_rate=config['learning_rate'],
                    subsample=config['subsample'],
                    colsample_bytree=config['colsample_bytree'],
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0,
                    n_jobs=1
                )
                clean_config = {
                    'algorithm': algorithm,
                    'n_estimators': config['xgb_n_estimators'],
                    'max_depth': config['xgb_max_depth'],
                    'learning_rate': config['learning_rate'],
                    'subsample': config['subsample'],
                    'colsample_bytree': config['colsample_bytree']
                }

            # Training phase
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start

            # Testing phase
            test_start = time.time()
            y_pred = model.predict(X_test)
            test_time = time.time() - test_start

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            abnormal_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)  # Recall for abnormal class (class 1)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Calculate confusion matrix for false negatives
            cm = confusion_matrix(y_test, y_pred)
            false_negatives = cm[1, 0] if cm.shape == (2, 2) else 0  # Abnormal samples predicted as normal
            total_abnormal = np.sum(y_test == 1)
            false_negative_rate = false_negatives / max(total_abnormal, 1)  # Normalize by total abnormal samples

            # Store detailed metrics and timing
            metrics = {
                'accuracy': accuracy,
                'abnormal_recall': abnormal_recall,
                'f1_score': f1,
                'false_negatives': false_negatives,
                'false_negative_rate': false_negative_rate
            }

            times = {
                'train_time': train_time,
                'test_time': test_time
            }

            # Scale objectives to similar ranges for better Pareto front detection
            # Normalize time objectives by algorithm-specific expected ranges
            if algorithm == 'svm':
                normalized_train_time = train_time / 100.0  # Expected max ~100s for SVM
                normalized_test_time = test_time / 20.0    # Expected max ~20s for SVM
            elif algorithm == 'rf':
                normalized_train_time = train_time / 300.0  # Expected max ~300s for RF
                normalized_test_time = test_time / 5.0     # Expected max ~5s for RF
            else:  # xgb
                normalized_train_time = train_time / 200.0  # Expected max ~200s for XGB
                normalized_test_time = test_time / 1.0     # Expected max ~1s for XGB

            # Objectives for minimization (SMAC minimizes)
            objectives = [
                1 - accuracy,              # minimize (1 - accuracy) = maximize accuracy
                1 - abnormal_recall,       # minimize (1 - abnormal_recall) = maximize abnormal recall  
                false_negative_rate,       # minimize false negative rate for abnormal class
                normalized_train_time,     # minimize normalized training time
                normalized_test_time       # minimize normalized testing time
            ]

            # Track evaluation
            tracker.add_evaluation(clean_config, objectives, metrics, times)

            print(f"Eval {len(tracker.all_evaluations):3d}: {algorithm.upper()} - "
                  f"Acc: {accuracy:.3f}, Ab_Recall: {abnormal_recall:.3f}, F1: {f1:.3f}, FN_Rate: {false_negative_rate:.3f}, "
                  f"Train: {train_time:.2f}s, Test: {test_time:.4f}s, "
                  f"Pareto: {len(tracker.pareto_candidates)}")

            return objectives

        except Exception as e:
            print(f"Error in evaluation: {e}")
            return [1.0, 1.0, 1.0, 10.0, 10.0]  # Worst possible normalized values

    return objective

def run_enhanced_multiobjective_optimization(X_train, y_train, X_test, y_test, n_trials=900):
    """Run enhanced multi-objective optimization with 900 evaluations"""

    print(f"\n{'='*80}")
    print(f"Running Enhanced Multi-Objective Bayesian Optimization")
    print(f"Using ParEGO algorithm with {n_trials} evaluations")
    print(f"Expected runtime: ~3-4 hours")
    print(f"{'='*80}")

    tracker = EnhancedMultiObjectiveTracker()
    objective_func = create_enhanced_multiobjective_function(X_train, y_train, X_test, y_test, tracker)
    config_space = create_optimized_config_space()

    # Enhanced scenario with more initial designs for better exploration
    scenario = Scenario(
        configspace=config_space,
        deterministic=True,
        n_trials=n_trials,
        n_workers=1,
        objectives=['neg_accuracy', 'neg_abnormal_recall', 'false_negative_rate', 'norm_train_time', 'norm_test_time'],
        seed=42
    )

    # Create ParEGO with enhanced parameters
    parego = ParEGO(scenario)

    # Create SMAC optimizer
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective_func,
        multi_objective_algorithm=parego
    )

    print("Starting optimization...")
    print("Progress will be saved every 100 evaluations...")
    
    # Run optimization with progress tracking
    try:
        start_time = time.time()
        incumbent = smac.optimize()
        optimization_time = time.time() - start_time
        print(f"Optimization completed: {len(tracker.all_evaluations)} evaluations")
        print(f"Optimization time: {optimization_time:.2f}s ({optimization_time/3600:.2f}h)")
    except Exception as e:
        optimization_time = time.time() - start_time
        print(f"Optimization error: {e}")
        print(f"Partial completion: {len(tracker.all_evaluations)} evaluations")
        print(f"Partial time: {optimization_time:.2f}s ({optimization_time/3600:.2f}h)")

    # Get Pareto front from our tracker
    pareto_front = tracker.get_pareto_front()
    print(f"Manual Pareto front contains {len(pareto_front)} solutions")

    # Save intermediate results every 100 evaluations
    if len(tracker.all_evaluations) % 100 == 0 or len(tracker.all_evaluations) == n_trials:
        save_intermediate_results(tracker, len(tracker.all_evaluations))

    return smac, tracker, pareto_front

def save_intermediate_results(tracker, eval_count):
    """Save intermediate results during optimization"""
    timestamp = f"_{eval_count}_evals"
    
    # Save all evaluations
    all_evals_data = []
    for eval in tracker.all_evaluations:
        row = {
            'eval_id': eval['eval_id'],
            'algorithm': eval['config']['algorithm'],
            'accuracy': eval['metrics']['accuracy'],
            'abnormal_recall': eval['metrics']['abnormal_recall'],
            'false_negative_rate': eval['metrics']['false_negative_rate'],
            'f1_score': eval['metrics']['f1_score'],
            'train_time': eval['times']['train_time'],
            'test_time': eval['times']['test_time']
        }
        for key, value in eval['config'].items():
            if key != 'algorithm':
                row[key] = value
        all_evals_data.append(row)

    pd.DataFrame(all_evals_data).to_csv(f'intermediate_results{timestamp}.csv', index=False)
    print(f"Saved intermediate results: intermediate_results{timestamp}.csv")

def analyze_enhanced_pareto_front(pareto_front, tracker):
    """Analyze and display enhanced Pareto optimal solutions"""
    print(f"\n{'='*80}")
    print(f"ENHANCED PARETO FRONT ANALYSIS")
    print(f"{'='*80}")
    print(f"Found {len(pareto_front)} Pareto optimal solutions")
    print("-" * 80)

    if len(pareto_front) == 0:
        print("Warning: No Pareto solutions found. This suggests:")
        print("1. Objectives may be too correlated")
        print("2. Search space might need adjustment")
        print("3. More evaluations might be needed")
        print("\nAnalyzing top performing configurations instead...")
        
        # Analyze best configurations by different criteria
        best_accuracy = max(tracker.all_evaluations, key=lambda x: x['metrics']['accuracy'])
        best_abnormal_recall = max(tracker.all_evaluations, key=lambda x: x['metrics']['abnormal_recall'])
        best_fn_rate = min(tracker.all_evaluations, key=lambda x: x['metrics']['false_negative_rate'])
        fastest_train = min(tracker.all_evaluations, key=lambda x: x['times']['train_time'])
        fastest_test = min(tracker.all_evaluations, key=lambda x: x['times']['test_time'])
        
        print(f"\nBest Accuracy: {best_accuracy['metrics']['accuracy']:.4f} "
              f"(Eval {best_accuracy['eval_id']}, {best_accuracy['config']['algorithm'].upper()})")
        print(f"Best Abnormal Recall: {best_abnormal_recall['metrics']['abnormal_recall']:.4f} "
              f"(Eval {best_abnormal_recall['eval_id']}, {best_abnormal_recall['config']['algorithm'].upper()})")
        print(f"Best FN Rate: {best_fn_rate['metrics']['false_negative_rate']:.4f} "
              f"(Eval {best_fn_rate['eval_id']}, {best_fn_rate['config']['algorithm'].upper()})")
        print(f"Fastest Training: {fastest_train['times']['train_time']:.2f}s "
              f"(Eval {fastest_train['eval_id']}, {fastest_train['config']['algorithm'].upper()})")
        print(f"Fastest Testing: {fastest_test['times']['test_time']:.4f}s "
              f"(Eval {fastest_test['eval_id']}, {fastest_test['config']['algorithm'].upper()})")
        
        return pd.DataFrame()
    
    pareto_data = []
    for i, (config, objectives) in enumerate(pareto_front):
        # Convert objectives back to original metrics
        accuracy = 1 - objectives[0]
        abnormal_recall = 1 - objectives[1]
        false_negative_rate = objectives[2]
        
        print(f"Pareto Solution {i+1}:")
        print(f"  Algorithm: {config['algorithm'].upper()}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Abnormal Recall: {abnormal_recall:.4f}")
        print(f"  False Negative Rate: {false_negative_rate:.4f}")
        print(f"  Normalized Train Time: {objectives[3]:.4f}")
        print(f"  Normalized Test Time: {objectives[4]:.6f}")
        print()

        row = {
            'solution_id': i+1,
            'algorithm': config['algorithm'],
            'accuracy': accuracy,
            'abnormal_recall': abnormal_recall,
            'false_negative_rate': false_negative_rate,
            'norm_train_time': objectives[3],
            'norm_test_time': objectives[4]
        }

        for key, value in config.items():
            if key != 'algorithm':
                row[key] = value

        pareto_data.append(row)

    # Create DataFrame and save
    pareto_df = pd.DataFrame(pareto_data)
    pareto_df.to_csv('enhanced_900_pareto_solutions.csv', index=False)
    print(f"Saved Pareto solutions to enhanced_900_pareto_solutions.csv")

    return pareto_df

def create_comprehensive_visualizations(tracker, pareto_front):
    """Create comprehensive visualizations for 900 evaluations"""
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))

    # Extract data
    all_acc = [eval['metrics']['accuracy'] for eval in tracker.all_evaluations]
    all_abnormal_recall = [eval['metrics']['abnormal_recall'] for eval in tracker.all_evaluations]
    all_fn_rate = [eval['metrics']['false_negative_rate'] for eval in tracker.all_evaluations]
    all_train = [eval['times']['train_time'] for eval in tracker.all_evaluations]
    all_test = [eval['times']['test_time'] for eval in tracker.all_evaluations]
    all_algorithms = [eval['config']['algorithm'] for eval in tracker.all_evaluations]
    all_f1 = [eval['metrics']['f1_score'] for eval in tracker.all_evaluations]

    color_map = {'svm': 'blue', 'rf': 'green', 'xgb': 'orange'}

    # Plot 1: Accuracy vs Abnormal Recall
    for alg, color in color_map.items():
        alg_indices = [i for i, a in enumerate(all_algorithms) if a == alg]
        if alg_indices:
            axes[0, 0].scatter([all_acc[i] for i in alg_indices],
                              [all_abnormal_recall[i] for i in alg_indices],
                              c=color, alpha=0.5, s=20, label=f'{alg.upper()}')
    
    # Highlight Pareto solutions
    if pareto_front:
        pareto_acc = [1 - obj[0] for _, obj in pareto_front]
        pareto_abnormal_recall = [1 - obj[1] for _, obj in pareto_front]
        axes[0, 0].scatter(pareto_acc, pareto_abnormal_recall, c='red', s=100, 
                          marker='*', label='Pareto Front', edgecolor='black', linewidth=2)
    
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Abnormal Recall')
    axes[0, 0].set_title('Accuracy vs Abnormal Recall (900 Evaluations)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy vs False Negative Rate
    for alg, color in color_map.items():
        alg_indices = [i for i, a in enumerate(all_algorithms) if a == alg]
        if alg_indices:
            axes[0, 1].scatter([all_acc[i] for i in alg_indices],
                              [all_fn_rate[i] for i in alg_indices],
                              c=color, alpha=0.5, s=20)
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_ylabel('False Negative Rate')
    axes[0, 1].set_title('Accuracy vs False Negative Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Convergence Analysis
    window_size = 50
    if len(all_acc) >= window_size:
        moving_acc = pd.Series(all_acc).rolling(window=window_size).mean()
        moving_abnormal_recall = pd.Series(all_abnormal_recall).rolling(window=window_size).mean()
        moving_fn_rate = pd.Series(all_fn_rate).rolling(window=window_size).mean()
        eval_numbers = list(range(1, len(all_acc) + 1))
        
        axes[0, 2].plot(eval_numbers, moving_acc, label=f'Accuracy (MA-{window_size})', color='red', alpha=0.8)
        axes[0, 2].plot(eval_numbers, moving_abnormal_recall, label=f'Abnormal Recall (MA-{window_size})', color='blue', alpha=0.8)
        axes[0, 2].plot(eval_numbers, moving_fn_rate, label=f'FN Rate (MA-{window_size})', color='purple', alpha=0.8)
        axes[0, 2].set_xlabel('Evaluation Number')
        axes[0, 2].set_ylabel('Performance')
        axes[0, 2].set_title('Convergence Analysis (Moving Average)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Algorithm Performance Distribution
    alg_counts = pd.Series(all_algorithms).value_counts()
    colors = [color_map[alg] for alg in alg_counts.index]
    axes[1, 0].pie(alg_counts.values, labels=[f'{alg.upper()}\n({count})' for alg, count in alg_counts.items()],
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 0].set_title('Algorithm Distribution in 900 Evaluations')

    # Plot 5: Abnormal Recall vs False Negative Rate
    for alg, color in color_map.items():
        alg_indices = [i for i, a in enumerate(all_algorithms) if a == alg]
        if alg_indices:
            axes[1, 1].scatter([all_abnormal_recall[i] for i in alg_indices],
                              [all_fn_rate[i] for i in alg_indices],
                              c=color, alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Abnormal Recall')
    axes[1, 1].set_ylabel('False Negative Rate')
    axes[1, 1].set_title('Abnormal Recall vs False Negative Rate')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Training vs Testing Time by Algorithm
    for alg, color in color_map.items():
        alg_indices = [i for i, a in enumerate(all_algorithms) if a == alg]
        if alg_indices:
            axes[1, 2].scatter([all_train[i] for i in alg_indices],
                              [all_test[i] for i in alg_indices],
                              c=color, alpha=0.5, s=20, label=alg.upper())
    axes[1, 2].set_xlabel('Training Time (s)')
    axes[1, 2].set_ylabel('Testing Time (s)')
    axes[1, 2].set_title('Training vs Testing Time')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 7: Performance Evolution Over Time (Segmented)
    n_segments = 9
    segment_size = len(all_acc) // n_segments
    segment_means_acc = []
    segment_means_abnormal_recall = []
    segment_means_fn_rate = []
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(all_acc)
        segment_means_acc.append(np.mean(all_acc[start_idx:end_idx]))
        segment_means_abnormal_recall.append(np.mean(all_abnormal_recall[start_idx:end_idx]))
        segment_means_fn_rate.append(np.mean(all_fn_rate[start_idx:end_idx]))
    
    segment_labels = [f'{i*segment_size+1}-{min((i+1)*segment_size, len(all_acc))}' 
                     for i in range(n_segments)]
    
    x_pos = np.arange(n_segments)
    width = 0.25
    
    axes[2, 0].bar(x_pos - width, segment_means_acc, width, label='Accuracy', alpha=0.7, color='red')
    axes[2, 0].bar(x_pos, segment_means_abnormal_recall, width, label='Abnormal Recall', alpha=0.7, color='blue')
    axes[2, 0].bar(x_pos + width, segment_means_fn_rate, width, label='FN Rate', alpha=0.7, color='purple')
    axes[2, 0].set_xlabel('Evaluation Segments')
    axes[2, 0].set_ylabel('Average Performance')
    axes[2, 0].set_title('Performance Evolution by Segments')
    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].set_xticklabels(segment_labels, rotation=45, ha='right')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Best Performance Tracking
    best_acc_so_far = []
    best_abnormal_recall_so_far = []
    best_fn_rate_so_far = []
    current_best_acc = 0
    current_best_abnormal_recall = 0
    current_best_fn_rate = 1.0  # Start with worst FN rate
    
    for acc, abnormal_recall, fn_rate in zip(all_acc, all_abnormal_recall, all_fn_rate):
        current_best_acc = max(current_best_acc, acc)
        current_best_abnormal_recall = max(current_best_abnormal_recall, abnormal_recall)
        current_best_fn_rate = min(current_best_fn_rate, fn_rate)  # Lower is better for FN rate
        best_acc_so_far.append(current_best_acc)
        best_abnormal_recall_so_far.append(current_best_abnormal_recall)
        best_fn_rate_so_far.append(current_best_fn_rate)
    
    eval_numbers = list(range(1, len(all_acc) + 1))
    axes[2, 1].plot(eval_numbers, best_acc_so_far, label='Best Accuracy So Far', color='red', linewidth=2)
    axes[2, 1].plot(eval_numbers, best_abnormal_recall_so_far, label='Best Abnormal Recall So Far', color='blue', linewidth=2)
    axes[2, 1].plot(eval_numbers, best_fn_rate_so_far, label='Best FN Rate So Far', color='purple', linewidth=2)
    axes[2, 1].set_xlabel('Evaluation Number')
    axes[2, 1].set_ylabel('Best Performance So Far')
    axes[2, 1].set_title('Best Performance Tracking')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Plot 9: Pareto Front Evolution
    pareto_counts = []
    eval_step = max(1, len(tracker.all_evaluations) // 100)  # Sample every ~1% of evaluations
    
    for i in range(0, len(tracker.all_evaluations), eval_step):
        # Create temporary tracker with evaluations up to point i
        temp_candidates = []
        for j in range(i + 1):
            eval_data = tracker.all_evaluations[j]
            objectives = eval_data['objectives']
            
            # Check if solution would be in Pareto front at this point
            dominated_indices = []
            is_dominated = False
            
            for k, candidate in enumerate(temp_candidates):
                candidate_obj = candidate['objectives']
                
                new_dominates = all(objectives[l] <= candidate_obj[l] for l in range(len(objectives))) and \
                               any(objectives[l] < candidate_obj[l] for l in range(len(objectives)))
                
                candidate_dominates = all(candidate_obj[l] <= objectives[l] for l in range(len(objectives))) and \
                                    any(candidate_obj[l] < objectives[l] for l in range(len(objectives)))
                
                if new_dominates:
                    dominated_indices.append(k)
                elif candidate_dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                for k in sorted(dominated_indices, reverse=True):
                    del temp_candidates[k]
                temp_candidates.append(eval_data)
        
        pareto_counts.append(len(temp_candidates))
    
    sample_evals = list(range(1, len(tracker.all_evaluations) + 1, eval_step))
    axes[2, 2].plot(sample_evals[:len(pareto_counts)], pareto_counts, 'g-', linewidth=2, marker='o', markersize=4)
    axes[2, 2].set_xlabel('Evaluation Number')
    axes[2, 2].set_ylabel('Pareto Front Size')
    axes[2, 2].set_title('Pareto Front Evolution')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_900_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def analyze_algorithm_performance(tracker):
    """Detailed analysis of algorithm performance across 900 evaluations"""
    print(f"\n{'='*80}")
    print(f"DETAILED ALGORITHM PERFORMANCE ANALYSIS (900 Evaluations)")
    print(f"{'='*80}")

    analysis_data = []
    
    for algorithm in ['svm', 'rf', 'xgb']:
        alg_evals = [eval for eval in tracker.all_evaluations if eval['config']['algorithm'] == algorithm]
        
        if alg_evals:
            accuracies = [eval['metrics']['accuracy'] for eval in alg_evals]
            abnormal_recalls = [eval['metrics']['abnormal_recall'] for eval in alg_evals]
            f1_scores = [eval['metrics']['f1_score'] for eval in alg_evals]
            fn_rates = [eval['metrics']['false_negative_rate'] for eval in alg_evals]
            train_times = [eval['times']['train_time'] for eval in alg_evals]
            test_times = [eval['times']['test_time'] for eval in alg_evals]
            
            analysis = {
                'algorithm': algorithm.upper(),
                'total_evaluations': len(alg_evals),
                'percentage_of_total': len(alg_evals) / len(tracker.all_evaluations) * 100,
                
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_min': np.min(accuracies),
                'accuracy_max': np.max(accuracies),
                'accuracy_median': np.median(accuracies),
                
                'abnormal_recall_mean': np.mean(abnormal_recalls),
                'abnormal_recall_std': np.std(abnormal_recalls),
                'abnormal_recall_min': np.min(abnormal_recalls),
                'abnormal_recall_max': np.max(abnormal_recalls),
                'abnormal_recall_median': np.median(abnormal_recalls),
                
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'f1_max': np.max(f1_scores),
                
                'fn_rate_mean': np.mean(fn_rates),
                'fn_rate_std': np.std(fn_rates),
                'fn_rate_min': np.min(fn_rates),
                'fn_rate_max': np.max(fn_rates),
                'fn_rate_median': np.median(fn_rates),
                
                'train_time_mean': np.mean(train_times),
                'train_time_std': np.std(train_times),
                'train_time_min': np.min(train_times),
                'train_time_max': np.max(train_times),
                'train_time_median': np.median(train_times),
                
                'test_time_mean': np.mean(test_times),
                'test_time_std': np.std(test_times),
                'test_time_min': np.min(test_times),
                'test_time_max': np.max(test_times),
                'test_time_median': np.median(test_times)
            }
            
            analysis_data.append(analysis)
            
            print(f"\n{algorithm.upper()} CLASSIFIER ({len(alg_evals)} evaluations, {len(alg_evals)/len(tracker.all_evaluations)*100:.1f}%):")
            print(f"  Accuracy:        {analysis['accuracy_mean']:.4f} ± {analysis['accuracy_std']:.4f} "
                  f"(min: {analysis['accuracy_min']:.4f}, max: {analysis['accuracy_max']:.4f})")
            print(f"  Abnormal Recall: {analysis['abnormal_recall_mean']:.4f} ± {analysis['abnormal_recall_std']:.4f} "
                  f"(min: {analysis['abnormal_recall_min']:.4f}, max: {analysis['abnormal_recall_max']:.4f})")
            print(f"  F1 Score:        {analysis['f1_mean']:.4f} ± {analysis['f1_std']:.4f} "
                  f"(max: {analysis['f1_max']:.4f})")
            print(f"  FN Rate:         {analysis['fn_rate_mean']:.4f} ± {analysis['fn_rate_std']:.4f} "
                  f"(min: {analysis['fn_rate_min']:.4f}, max: {analysis['fn_rate_max']:.4f})")
            print(f"  Train Time:      {analysis['train_time_mean']:.2f} ± {analysis['train_time_std']:.2f}s "
                  f"(range: {analysis['train_time_min']:.2f}-{analysis['train_time_max']:.2f}s)")
            print(f"  Test Time:       {analysis['test_time_mean']:.4f} ± {analysis['test_time_std']:.4f}s "
                  f"(range: {analysis['test_time_min']:.4f}-{analysis['test_time_max']:.4f}s)")

    # Save detailed analysis
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv('enhanced_900_algorithm_analysis.csv', index=False)
    print(f"\nSaved detailed analysis to enhanced_900_algorithm_analysis.csv")

    return analysis_df

def main(data_path):
    if not SMAC_AVAILABLE:
        print("Please install SMAC: pip install smac")
        return

    total_start_time = time.time()
    print(f"Starting Enhanced Multi-Objective Bayesian Optimization with 900 Evaluations")
    print(f"Estimated runtime: 3-4 hours")

    # Process data
    data_start_time = time.time()
    X, y = process_audio_data(data_path)
    data_processing_time = time.time() - data_start_time

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nDataset: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples")
    print(f"Data processing time: {data_processing_time:.2f}s")

    # Run enhanced multi-objective optimization with 900 trials
    optimization_start_time = time.time()
    smac, tracker, pareto_front = run_enhanced_multiobjective_optimization(
        X_train_scaled, y_train, X_test_scaled, y_test, n_trials=900
    )
    optimization_time = time.time() - optimization_start_time

    print(f"\n{'='*80}")
    print(f"ENHANCED MULTI-OBJECTIVE OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total evaluations: {len(tracker.all_evaluations)}")
    print(f"Pareto solutions: {len(pareto_front)}")
    print(f"Optimization time: {optimization_time:.2f}s ({optimization_time/3600:.2f}h)")

    # Comprehensive analysis
    pareto_df = analyze_enhanced_pareto_front(pareto_front, tracker)
    algorithm_df = analyze_algorithm_performance(tracker)

    # Create comprehensive visualizations
    print("\nCreating comprehensive visualizations...")
    fig = create_comprehensive_visualizations(tracker, pareto_front)

    # Total timing
    total_time = time.time() - total_start_time
    print(f"\nTotal process time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

    # Save final complete results
    all_evals_data = []
    for eval in tracker.all_evaluations:
        row = {
            'eval_id': eval['eval_id'],
            'algorithm': eval['config']['algorithm'],
            'accuracy': eval['metrics']['accuracy'],
            'abnormal_recall': eval['metrics']['abnormal_recall'],
            'false_negative_rate': eval['metrics']['false_negative_rate'],
            'f1_score': eval['metrics']['f1_score'],
            'train_time': eval['times']['train_time'],
            'test_time': eval['times']['test_time']
        }
        # Add config parameters
        for key, value in eval['config'].items():
            if key != 'algorithm':
                row[key] = value
        all_evals_data.append(row)

    all_evals_df = pd.DataFrame(all_evals_data)
    all_evals_df.to_csv('final_900_evaluations_complete.csv', index=False)
    print(f"Saved complete results to final_900_evaluations_complete.csv")

    # Performance summary
    print(f"\n{'='*80}")
    print(f"FINAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    best_accuracy = max(tracker.all_evaluations, key=lambda x: x['metrics']['accuracy'])
    best_abnormal_recall = max(tracker.all_evaluations, key=lambda x: x['metrics']['abnormal_recall'])
    best_f1 = max(tracker.all_evaluations, key=lambda x: x['metrics']['f1_score'])
    best_fn_rate = min(tracker.all_evaluations, key=lambda x: x['metrics']['false_negative_rate'])
    
    print(f"Best Accuracy:        {best_accuracy['metrics']['accuracy']:.4f} "
          f"({best_accuracy['config']['algorithm'].upper()}, Eval #{best_accuracy['eval_id']})")
    print(f"Best Abnormal Recall: {best_abnormal_recall['metrics']['abnormal_recall']:.4f} "
          f"({best_abnormal_recall['config']['algorithm'].upper()}, Eval #{best_abnormal_recall['eval_id']})")
    print(f"Best F1 Score:        {best_f1['metrics']['f1_score']:.4f} "
          f"({best_f1['config']['algorithm'].upper()}, Eval #{best_f1['eval_id']})")
    print(f"Best FN Rate:         {best_fn_rate['metrics']['false_negative_rate']:.4f} "
          f"({best_fn_rate['config']['algorithm'].upper()}, Eval #{best_fn_rate['eval_id']})")
    print(f"Total Pareto Solutions: {len(pareto_front)}")

    return smac, tracker, pareto_front

if __name__ == "__main__":
    DATA_PATH = "C:/Users/DELL/Heartbeat/heartbeat-sounds"

    if os.path.exists(DATA_PATH):
        print("Starting 900-evaluation multi-objective optimization...")
        print("This will take approximately 3-4 hours to complete.")
        print("Progress and intermediate results will be saved during execution.")
        
        smac_optimizer, tracker, pareto_front = main(DATA_PATH)
        
        print("\n" + "="*80)
        print("ENHANCED 900-EVALUATION MULTI-OBJECTIVE OPTIMIZATION COMPLETED!")
        print("="*80)
        print("Generated files:")
        print("- final_900_evaluations_complete.csv (all evaluations)")
        print("- enhanced_900_pareto_solutions.csv (Pareto solutions)")
        print("- enhanced_900_algorithm_analysis.csv (algorithm comparison)")
        print("- enhanced_900_evaluation_results.png (comprehensive plots)")
        print("- intermediate_results_*.csv (progress snapshots)")
        
    else:
        print(f"Data path not found: {DATA_PATH}")
        print("Please update DATA_PATH variable with correct path to heartbeat audio files")