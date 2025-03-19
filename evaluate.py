# evaluate.py
import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import librosa
import torch.nn.functional as F
import seaborn as sns

from config import *
from data_utils import prepare_dataloader, load_audio
from augmented_dataset import standardize_audio_length, prepare_augmented_dataloader
from models.fast_classifier import FastIntentClassifier
from models.precise_classifier import PreciseIntentClassifier
from inference import IntentInferenceEngine

def evaluate_models(data_dir, annotation_file, fast_model_path, precise_model_path=None, output_dir='evaluation_results', analyze_length_impact=False):
    """Evaluate model performance"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference engine
    inference_engine = IntentInferenceEngine(
        fast_model_path=fast_model_path,
        precise_model_path=precise_model_path
    )
    
    # Define feature extraction function
    def fast_feature_extractor(audio, sr, **kwargs):
        audio = standardize_audio_length(audio, sr)
        # Extract MFCC features and dynamic features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        return features.T  # (time, features) format, without context
    
    # Also perform audio length analysis
    fast_loader, fast_labels = prepare_augmented_dataloader(
        annotation_file=annotation_file, 
        data_dir=data_dir, 
        feature_extractor=fast_feature_extractor,
        batch_size=32, 
        augment=False,  # No augmentation during evaluation
        shuffle=False
    )
    
    def precise_feature_extractor(audio, sr, transcript=None, **kwargs):
        # For precise classifier, we need text features
        # In actual use, we should use ASR to get text
        # But in evaluation, we use annotated text
        if transcript is None:
            transcript = "Default text for evaluation mode"
            
        # Use tokenizer to process text
        encoding = inference_engine.tokenizer(
            transcript,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    # Prepare data loader for precise classifier
    if precise_model_path:
        precise_loader, precise_labels = prepare_augmented_dataloader(
            annotation_file=annotation_file, 
            data_dir=data_dir, 
            feature_extractor=precise_feature_extractor,
            batch_size=32, 
            augment=False,
            shuffle=False
        )
    
    print("Preparing test data...")
    
    # Store evaluation results
    all_results = []
    
    # Evaluate fast classifier
    print("\nStarting fast classifier evaluation...")
    fast_results = evaluate_fast_model(inference_engine, fast_loader)
    all_results.append(fast_results)
    
    # Print fast classifier results
    print(f"\nFast Classifier Evaluation Results:")
    print(f"Accuracy: {fast_results['accuracy']:.4f}")
    print(f"Precision: {fast_results['precision']:.4f}")
    print(f"Recall: {fast_results['recall']:.4f}")
    print(f"F1 Score: {fast_results['f1']:.4f}")
    print(f"Average Inference Time: {fast_results['avg_inference_time_ms']:.2f}ms")
    print("\nClassification Report:")
    print(fast_results['class_report'])
    
    # Evaluate precise classifier (if available)
    if precise_model_path:
        print("\nStarting precise classifier evaluation...")
        precise_results = evaluate_precise_model(inference_engine, precise_loader)
        all_results.append(precise_results)
        
        # Print precise classifier results
        print(f"\nPrecise Classifier Evaluation Results:")
        print(f"Accuracy: {precise_results['accuracy']:.4f}")
        print(f"Precision: {precise_results['precision']:.4f}")
        print(f"Recall: {precise_results['recall']:.4f}")
        print(f"F1 Score: {precise_results['f1']:.4f}")
        print(f"Average Inference Time: {precise_results['avg_inference_time_ms']:.2f}ms")
        print("\nClassification Report:")
        print(precise_results['class_report'])
        
        # Evaluate full pipeline
        print("\nStarting full inference pipeline evaluation...")
        pipeline_results = evaluate_full_pipeline(inference_engine, fast_loader)
        all_results.append(pipeline_results)
        
        # Print pipeline results
        print(f"\nFull Inference Pipeline Evaluation Results:")
        print(f"Accuracy: {pipeline_results['accuracy']:.4f}")
        print(f"Precision: {pipeline_results['precision']:.4f}")
        print(f"Recall: {pipeline_results['recall']:.4f}")
        print(f"F1 Score: {pipeline_results['f1']:.4f}")
        print(f"Average Inference Time: {pipeline_results['avg_inference_time_ms']:.2f}ms")
        print(f"Fast Classifier Usage Rate: {pipeline_results['fast_ratio']*100:.1f}%")
        print("\nClassification Report:")
        print(pipeline_results['class_report'])
    
    # Analyze impact of audio length on performance
    length_impact_results = None
    if analyze_length_impact:
        print("\nAnalyzing impact of audio length on performance...")
        length_impact_results = analyze_audio_length_impact(inference_engine, data_dir, annotation_file)
    
    # Generate visualization
    print("\nGenerating evaluation result visualizations...")
    generate_visualizations(all_results, output_dir, timestamp, length_impact_results)
    
    print(f"\nEvaluation complete. Results saved to {output_dir} directory.")
    
    return all_results, length_impact_results

def evaluate_fast_model(inference_engine, data_loader):
    """Evaluate fast classifier model performance"""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_inference_times = []

    loop = tqdm(data_loader, desc="Evaluating fast classifier")
    for batch_idx, batch in enumerate(loop):
        features, labels = batch['features'], batch['label']
        
        for feature, label in zip(features, labels):
            # Single sample inference
            feature_np = feature.numpy() if not feature.is_cuda else feature.cpu().numpy()
            
            start_time = time.time()
            with torch.no_grad():
                predicted_class, confidence = inference_engine.fast_model.predict(
                    torch.FloatTensor(feature_np).unsqueeze(0).to(inference_engine.device)
                )
            inference_time = time.time() - start_time
            
            # Collect results
            all_predictions.append(predicted_class.item())
            all_confidences.append(confidence.item())
            all_labels.append(label.item() if torch.is_tensor(label) else label)
            all_inference_times.append(inference_time)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    avg_inference_time = np.mean(all_inference_times) * 1000  # Convert to milliseconds
    
    # Generate more detailed report
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=INTENT_CLASSES, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'model_type': 'fast',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_report': class_report,
        'confusion_matrix': cm,
        'labels': INTENT_CLASSES,
        'all_predictions': all_predictions,
        'all_confidences': all_confidences,
        'all_labels': all_labels,
        'all_inference_times': all_inference_times
    }
    
    return results

def evaluate_precise_model(inference_engine, data_loader):
    """Evaluate precise classifier model performance"""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_inference_times = []

    loop = tqdm(data_loader, desc="Evaluating precise classifier")
    for batch_idx, batch in enumerate(loop):
        features = batch['features']
        labels = batch['label']
        
        for i in range(len(labels)):
            input_ids = features['input_ids'][i].unsqueeze(0).to(inference_engine.device)
            attention_mask = features['attention_mask'][i].unsqueeze(0).to(inference_engine.device)
            label = labels[i]
            
            # Single sample inference
            start_time = time.time()
            with torch.no_grad():
                outputs = inference_engine.precise_model(input_ids, attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
            inference_time = time.time() - start_time
            
            # Collect results
            all_predictions.append(predicted.item())
            all_confidences.append(confidence.item())
            all_labels.append(label.item() if torch.is_tensor(label) else label)
            all_inference_times.append(inference_time)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    avg_inference_time = np.mean(all_inference_times) * 1000  # Convert to milliseconds
    
    # Generate more detailed report
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=INTENT_CLASSES, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'model_type': 'precise',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_report': class_report,
        'confusion_matrix': cm,
        'labels': INTENT_CLASSES,
        'all_predictions': all_predictions,
        'all_confidences': all_confidences,
        'all_labels': all_labels,
        'all_inference_times': all_inference_times
    }
    
    return results

def evaluate_full_pipeline(inference_engine, data_loader, confidence_threshold=FAST_CONFIDENCE_THRESHOLD):
    """Evaluate full inference pipeline performance (first-level fast + second-level precise classifier)"""
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_inference_times = []
    all_is_fast = []  # Record whether fast classifier was used

    # Use passed threshold instead of engine default
    original_threshold = inference_engine.fast_confidence_threshold
    inference_engine.fast_confidence_threshold = confidence_threshold
    
    loop = tqdm(data_loader, desc="Evaluating full pipeline")
    for batch_idx, batch in enumerate(loop):
        features = batch['features']
        labels = batch['label']
        
        for i in range(len(labels)):
            # Prepare features
            if isinstance(features, torch.Tensor):
                # For fast classifier features
                feature = features[i]
                feature_np = feature.cpu().numpy() if feature.is_cuda else feature.numpy()
                feature_tensor = torch.FloatTensor(feature_np).unsqueeze(0).to(inference_engine.device)
                label = labels[i]
                transcription = None
            elif isinstance(features, dict):
                # For precise classifier features
                feature_tensor = {
                    'input_ids': features['input_ids'][i].unsqueeze(0).to(inference_engine.device),
                    'attention_mask': features['attention_mask'][i].unsqueeze(0).to(inference_engine.device)
                }
                label = labels[i]
                transcription = batch.get('transcription', [None])[i]
            else:
                print(f"Unsupported feature type: {type(features)}")
                continue
            
            # Single sample inference
            start_time = time.time()
            with torch.no_grad():
                # Use predict method directly on features
                predicted_class, confidence, is_fast = inference_engine.predict(feature_tensor)
            inference_time = time.time() - start_time
            
            # Collect results
            all_predictions.append(predicted_class)
            all_confidences.append(confidence)
            all_labels.append(label.item() if torch.is_tensor(label) else label)
            all_inference_times.append(inference_time)
            all_is_fast.append(is_fast)

    # Restore original threshold
    inference_engine.fast_confidence_threshold = original_threshold
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    avg_inference_time = np.mean(all_inference_times) * 1000  # Convert to milliseconds
    fast_ratio = sum(all_is_fast) / len(all_is_fast) if all_is_fast else 0
    
    # Generate more detailed report
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=INTENT_CLASSES, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'model_type': 'pipeline',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'class_report': class_report,
        'confusion_matrix': cm,
        'labels': INTENT_CLASSES,
        'all_predictions': all_predictions,
        'all_confidences': all_confidences,
        'all_labels': all_labels,
        'all_inference_times': all_inference_times,
        'all_is_fast': all_is_fast,
        'fast_ratio': fast_ratio
    }
    
    return results

def analyze_audio_length_impact(inference_engine, data_dir, annotation_file):
    """Analyze impact of audio length on model performance"""
    print("Loading dataset information...")
    # Read annotation file
    annotations = pd.read_csv(annotation_file)
    
    # Length grouping definition
    length_bins = {
        'Short (<=1s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []},
        'Medium (1-3s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []},
        'Long (3-5s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []},
        'Super Long (>5s)': {'samples': [], 'labels': [], 'predictions': [], 'times': []}
    }
    
    # Get class to index mapping
    class_to_idx = {cls: i for i, cls in enumerate(INTENT_CLASSES)}
    
    print("Analyzing different length audio samples...")
    # Iterate over dataset
    for idx in tqdm(range(len(annotations))):
        try:
            # Get audio file path and label
            audio_path = os.path.join(data_dir, annotations.iloc[idx]['file_path'])
            intent_label = annotations.iloc[idx]['intent']
            label_idx = class_to_idx[intent_label]
            
            # Load audio file
            audio, sr = load_audio(audio_path)
            duration = len(audio) / sr
            
            # Decide which length group audio belongs to
            if duration <= 1.0:
                group = 'Short (<=1s)'
            elif duration <= 3.0:
                group = 'Medium (1-3s)'
            elif duration <= 5.0:
                group = 'Long (3-5s)'
            else:
                group = 'Super Long (>5s)'
            
            # Preprocess audio (optional, depending on whether to evaluate original audio or preprocessed audio)
            audio = standardize_audio_length(audio, sr)
            
            # Extract features and predict
            start_time = time.time()
            features = inference_engine.preprocessor.process(audio)
            features = inference_engine.feature_extractor.extract_features(features)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(inference_engine.device)
            
            with torch.no_grad():
                prediction, confidence = inference_engine.fast_model.predict(features_tensor)
            
            inference_time = time.time() - start_time
            
            # Record results
            length_bins[group]['samples'].append(idx)
            length_bins[group]['labels'].append(label_idx)
            length_bins[group]['predictions'].append(prediction.item())
            length_bins[group]['times'].append(inference_time)
        
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    # Calculate performance metrics for each length group
    results = []
    
    for group, data in length_bins.items():
        if len(data['samples']) > 0:
            accuracy = accuracy_score(data['labels'], data['predictions'])
            precision, recall, f1, _ = precision_recall_fscore_support(
                data['labels'], data['predictions'], average='macro'
            )
            avg_time = np.mean(data['times']) * 1000  # Convert to milliseconds
            
            results.append({
                'Length Group': group,
                'Sample Count': len(data['samples']),
                'Accuracy': accuracy,
                'Precision (Macro Average)': precision,
                'Recall (Macro Average)': recall,
                'F1 Score (Macro Average)': f1,
                'Average Inference Time (ms)': avg_time
            })
            
            print(f"\n{group} Group Evaluation Results (Sample Count: {len(data['samples'])}):")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (Macro Average): {precision:.4f}")
            print(f"Recall (Macro Average): {recall:.4f}")
            print(f"F1 Score (Macro Average): {f1:.4f}")
            print(f"Average Inference Time: {avg_time:.2f}ms")
    
    return results

def generate_visualizations(results, output_dir, timestamp, length_impact_results=None):
    """Generate visualizations for evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create accuracy comparison chart for all models
    plt.figure(figsize=(10, 6))
    models = [result['model_type'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    
    plt.bar(models, accuracies, color=['blue', 'green', 'red'][:len(models)])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{timestamp}_accuracy_comparison.png'))
    
    # Create inference time comparison chart
    plt.figure(figsize=(10, 6))
    inference_times = [result['avg_inference_time_ms'] for result in results]
    
    plt.bar(models, inference_times, color=['blue', 'green', 'red'][:len(models)])
    plt.title('Model Inference Time Comparison (ms)')
    plt.xlabel('Model')
    plt.ylabel('Average Inference Time (ms)')
    
    for i, v in enumerate(inference_times):
        plt.text(i, v + 0.5, f'{v:.2f}ms', ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{timestamp}_inference_time_comparison.png'))
    
    # Generate confusion matrix for each model
    for result in results:
        plt.figure(figsize=(10, 8))
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=INTENT_CLASSES,
                   yticklabels=INTENT_CLASSES)
        
        plt.title(f'{result["model_type"]} Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{timestamp}_{result["model_type"]}_confusion_matrix.png'))
    
    # Generate inference time distribution for each model
    for result in results:
        plt.figure(figsize=(10, 6))
        inference_times = np.array(result['all_inference_times']) * 1000  # Convert to milliseconds
        plt.hist(inference_times, bins=30, alpha=0.7, color='blue')
        plt.title(f'{result["model_type"]} Inference Time Distribution')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Sample Count')
        plt.savefig(os.path.join(output_dir, f'{timestamp}_{result["model_type"]}_inference_time_distribution.png'))
    
    # If there are pipeline results, also show the proportion of second-level model use
    for result in results:
        if result['model_type'] == 'pipeline' and 'all_is_fast' in result:
            plt.figure(figsize=(8, 8))
            fast_count = sum(result['all_is_fast'])
            precise_count = len(result['all_is_fast']) - fast_count
            
            plt.pie([fast_count, precise_count], 
                   labels=['Fast Model', 'Precise Model'],
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightgreen'])
            
            plt.title('Inference Path Distribution')
            plt.savefig(os.path.join(output_dir, f'{timestamp}_pipeline_path_distribution.png'))
    
    # If there are audio length impact analysis results
    if length_impact_results:
        plt.figure(figsize=(12, 6))
        durations = length_impact_results['durations']
        accuracies = length_impact_results['accuracies']
        times = length_impact_results['times']
        
        # Create dual y-axis chart
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Audio Length (s)')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(durations, accuracies, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Inference Time (ms)', color=color)
        ax2.plot(durations, times, color=color, marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Impact of Audio Length on Accuracy and Inference Time')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{timestamp}_length_impact.png'))
        
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate speech intent recognition model')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='Annotation file path')
    parser.add_argument('--fast_model', type=str, required=True, help='First-level fast classifier path')
    parser.add_argument('--precise_model', type=str, help='Second-level precise classifier path (optional)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--analyze_length', action='store_true', help='Analyze impact of audio length on performance')
    args = parser.parse_args()
    
    # Evaluate model and save results
    evaluate_models(
        args.data_dir,
        args.annotation_file,
        args.fast_model,
        args.precise_model,
        args.output_dir,
        args.analyze_length
    ) 