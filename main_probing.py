#!/usr/bin/env python3
"""
State-of-the-Art Four Probing Systems Evaluator (Paper-Aligned)

Generates comprehensive evaluation table with longtail analysis:
- Supports specific model/system/layer combinations
- Outputs exact table format with frequent vs longtail breakdown  
- Handles multiple models: Gemma, MedGemma, Llama, Mistral, etc.
- Implements all four probing systems: LRC, Circular, MLP, Logistic
- Uses top-10 for error detection (aligned with arXiv:2507.12379)

Usage:
    python main_probing.py --system mlp --model google/gemma-3-4b-pt --subject-layers 8 --object-layers 16
    python main_new.py --system all --model google/medgemma-4b-pt --subject-layers 6,8 --object-layers 12,16
"""

import argparse
import os
import sys
import json
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import time
from tqdm import tqdm
from gpu_utils import GPUMemoryManager, get_gpu_memory_info, clear_gpu_memory, monitor_gpu_usage
import pandas as pd

# Add src_probe to path
sys.path.append(str(Path(__file__).parent / "src_probe"))

from src_probe import (
    LRCProbingSystem,
    CircularProbingSystem,
    MLPProbingSystem,
    LogisticProbingSystem,
    KnowledgeTriple,
    ProbeResults
)

# Import utilities
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.error("Transformers library not available. Please install: pip install transformers accelerate")
    TRANSFORMERS_AVAILABLE = False


class StateOfTheArtEvaluator:
    """
    State-of-the-art evaluator for knowledge triple probing.
    
    Features:
    - Exact table format with longtail analysis
    - Support for three target models
    - All four probing systems
    - Comprehensive metrics reporting
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        k_values: List[int] = None,
        random_seed: int = 42,
        cv_folds: int = 5,
        gpu_config: Dict = None
    ):
        self.model_name = model_name
        self.device = device
        self.k_values = k_values or [1, 5, 10]  # Changed from [1, 5, 10, 25] to match paper (arXiv:2507.12379)
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        
        # GPU optimization configuration for H100s
        self.gpu_config = gpu_config or {
            'batch_size': 64,                    # Large batch size for H100s
            'gradient_accumulation_steps': 2,    # Effective batch size = 128
            'max_memory_per_gpu': "90GB",        # H100 ~95GB, leave 5GB buffer
            'use_mixed_precision': True,         # FP16/BF16 mixed precision
            'dataloader_num_workers': 4,         # Parallel data loading
            'pin_memory': True,                  # Faster GPU transfers
            'prefetch_factor': 2,                # Prefetch batches
        }
        
        # Model configuration mapping
        self.model_configs = {
            "google/gemma-3-4b-pt": {"display_name": "Gemma 3B (Base)", "base_model": "gemma-3b"},
            "google/gemma-2-9b": {"display_name": "Gemma 2 9B", "base_model": "gemma-2-9b"},
            "google/medgemma-4b-pt": {"display_name": "MedGemma 4B (Pretrained)", "base_model": "gemma-4b"},
            "google/medgemma-4b-it": {"display_name": "MedGemma 4B (Instruction)", "base_model": "gemma-4b"},
            "OpenMeditron/Meditron3-Gemma2-9B": {"display_name": "Meditron3 Gemma2 9B", "base_model": "gemma-2-9b"},
            "meta-llama/Llama-3.1-8B": {"display_name": "Llama 3.1 8B (Base)", "base_model": "llama-3.1-8b"},
            "meta-llama/Llama-3.1-8B-Instruct": {"display_name": "Llama 3.1 8B (Instruct)", "base_model": "llama-3.1-8b"},
            "TsinghuaC3I/Llama-3.1-8B-UltraMedical": {"display_name": "Llama 3.1 8B (UltraMedical)", "base_model": "llama-3.1-8b"},
            "meta-llama/Meta-Llama-3-8B": {"display_name": "Meta Llama 3 8B", "base_model": "llama-3-8b"},
            "Henrychur/MMed-Llama-3-8B": {"display_name": "MMed Llama 3 8B", "base_model": "llama-3-8b"},
            "mistralai/Mistral-7B-Instruct-v0.1": {"display_name": "Mistral 7B Instruct v0.1", "base_model": "mistral-7b"},
            "BioMistral/BioMistral-7B": {"display_name": "BioMistral 7B", "base_model": "mistral-7b"}
        }
        
        # Set random seeds
        self._set_random_seeds()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.systems = {}
        
        # Setup logging
        self._setup_logging()
        
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/sota_evaluation_{int(time.time())}.log'),
                logging.StreamHandler()
            ]
        )
        
        logging.info("🚀 State-of-the-Art Probing Systems Evaluator")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Device: {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
        
        logging.info(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with single GPU allocation to avoid device conflicts
            if self.device == "cuda" or self.device.startswith("cuda:"):
                # Use specific GPU device to avoid multi-GPU conflicts
                gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map={"": gpu_id},  # Load entire model on specific GPU
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).to(self.device)
            
            self.model.eval()
            logging.info(f"✅ Model loaded successfully")
            
            # Clear GPU cache and monitor usage
            clear_gpu_memory()
            monitor_gpu_usage()
            logging.info(f"GPU memory cleared after model loading")
            
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            raise
    
    def load_knowledge_data(
        self, 
        max_samples_per_relation: int = None
    ) -> Tuple[Dict[str, List[KnowledgeTriple]], Dict[str, List[KnowledgeTriple]]]:
        """
        Load knowledge data organized by relation.
        
        Returns:
            Tuple of (train_data_by_relation, test_data_by_relation, object_to_index)
        """
        logging.info("📊 Loading knowledge triple data...")
        
        def load_and_organize_triples(file_path: str) -> Dict[str, List[KnowledgeTriple]]:
            """Load triples and organize by relation."""
            relation_data = {}
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle nested JSON structure with 'samples' array
                if isinstance(data, dict) and 'samples' in data:
                    items = data['samples']
                    logging.info(f"Found {len(items)} samples in {file_path}")
                elif isinstance(data, list):
                    items = data
                    logging.info(f"Found {len(items)} items in {file_path}")
                else:
                    logging.error(f"Unexpected JSON structure in {file_path}")
                    return {}
                
                for item in items:
                    # Handle different possible field names
                    if 'relation' in item:
                        relation = item['relation']
                    elif 'relationship' in item:
                        relation = item['relationship']
                    else:
                        logging.warning(f"No relation field found in item: {item}")
                        continue
                    
                    if relation not in relation_data:
                        relation_data[relation] = []
                    
                    # Handle different possible field names
                    subject = item.get('subject', item.get('subj', ''))
                    obj = item.get('object', item.get('obj', ''))
                    
                    if not subject or not obj:
                        logging.warning(f"Missing subject/object in item: {item}")
                        continue
                    
                    triple = KnowledgeTriple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        is_longtail=item.get('is_longtail', False),
                        cooccurrence_count=item.get('cooccurrence_count', 0)
                    )
                    relation_data[relation].append(triple)
                
                # Optionally sample data per relation if max_samples_per_relation is set
                if max_samples_per_relation is not None:
                    for relation in relation_data:
                        if len(relation_data[relation]) > max_samples_per_relation:
                            relation_data[relation] = random.sample(
                                relation_data[relation], max_samples_per_relation
                            )
                
                logging.info(f"Loaded {sum(len(triples) for triples in relation_data.values())} triples from {file_path}")
                return relation_data
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
                return {}
        
        # Load data
        train_path = "/home/avv533/knowlegde/bin/data/others/longtail_knowledge_dataset/train_combined.json"
        test_path = "/home/avv533/knowlegde/bin/data/others/longtail_knowledge_dataset/test_combined.json"
        
        train_data = load_and_organize_triples(train_path)
        test_data = load_and_organize_triples(test_path)
        
        logging.info(f"📊 Relations loaded: {list(train_data.keys())}")
        return train_data, test_data
    
    def initialize_system(self, system_name: str, vocab_size: int = 100):
        """Initialize a specific probing system."""
        available_systems = {
            'lrc': LRCProbingSystem,
            'circular': CircularProbingSystem,
            'mlp': MLPProbingSystem,
            'logistic': LogisticProbingSystem
        }
        
        if system_name not in available_systems:
            raise ValueError(f"Unknown system: {system_name}")
        
        logging.info(f"Initializing {system_name} system with vocab_size={vocab_size}...")
        
        try:
            if system_name == 'lrc':
                system = available_systems[system_name](
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    k_values=self.k_values,
                    output_dim=vocab_size
                )
            else:
                system = available_systems[system_name](
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    output_dim=vocab_size,  # Use actual vocabulary size
                    k_values=self.k_values
                )
            
            logging.info(f"✅ {system_name} system initialized with output_dim={vocab_size}")
            return system
            
        except Exception as e:
            logging.error(f"❌ Error initializing {system_name} system: {e}")
            raise
    
    def evaluate_system_on_relation(
        self,
        system_name: str,
        relation: str,
        train_triples: List[KnowledgeTriple],
        test_triples: List[KnowledgeTriple],
        subject_layer: int,
        object_layer: int
    ) -> Dict:
        """
        Evaluate a system on a specific relation.
        
        Returns results broken down by cooccurrence_count buckets:
        - ≤ 10, (10, 100], (100, 1000], > 1000
        """
        logging.info(f"🔬 Evaluating {system_name} on {relation} (layers {subject_layer}->{object_layer})")
        
        # Categorize test data by cooccurrence_count buckets
        cooccurrence_buckets = self._categorize_by_cooccurrence(test_triples)
        
        # Log bucket sizes
        bucket_sizes = {bucket: len(triples) for bucket, triples in cooccurrence_buckets.items()}
        logging.info(f"   📊 Test data by cooccurrence buckets: {bucket_sizes}")
        
        results = {}
        
        # Build relation-scoped vocabulary from train+test triples
        rel_objects = sorted({t.object for t in train_triples + test_triples})
        object_to_index = {obj: idx for idx, obj in enumerate(rel_objects)}
        if len(object_to_index) == 0:
            logging.error(f"❌ Empty vocabulary for relation '{relation}' (no objects in train+test after sampling). Skipping relation.")
            return {'error': 'empty_vocabulary'}
        
        try:
            # Initialize system with relation-specific vocab size
            system = self.initialize_system(system_name, vocab_size=len(object_to_index))
            
            # Extract training activations (few-shot)
            train_activations = system.extract_activations(
                train_triples, subject_layer, object_layer, use_few_shot=True,
                batch_size=getattr(self, 'prompt_batch_size', 16)
            )
            train_labels = self._create_labels(train_triples, object_to_index, system)
            
            # Train the system (pass object_to_index for external-label generation)
            train_metrics = system.train_system(
                train_triples, train_activations, train_labels,
                subject_layer=subject_layer, object_layer=object_layer,
                object_to_index=object_to_index,
                n_folds=self.cv_folds  # Pass the cv_folds parameter
            )
            
            # Evaluate on each cooccurrence bucket
            for bucket_name, bucket_triples in cooccurrence_buckets.items():
                if bucket_triples:
                    try:
                        logging.info(f"   🔍 Evaluating bucket '{bucket_name}' ({len(bucket_triples)} samples)")
                        bucket_results = self._evaluate_subset(
                            system, bucket_triples, object_to_index, subject_layer, object_layer
                        )
                        results[bucket_name] = {
                            'test_size': len(bucket_triples),
                            'metrics': bucket_results
                        }
                        logging.info(f"   ✅ Bucket '{bucket_name}' evaluation successful")
                    except Exception as e:
                        logging.error(f"   ❌ Bucket '{bucket_name}' evaluation failed: {e}")
                        results[bucket_name] = {'test_size': len(bucket_triples), 'metrics': None, 'error': str(e)}
                else:
                    results[bucket_name] = {'test_size': 0, 'metrics': None}
            
            results['train_metrics'] = train_metrics
            results['total_test_size'] = len(test_triples)
            
        except Exception as e:
            logging.error(f"❌ Error evaluating {system_name} on {relation}: {e}")
            results['error'] = str(e)
        
        return results
    
    def evaluate_system_mixed_training(
        self,
        system_name: str,
        train_data: Dict[str, List[KnowledgeTriple]],
        test_data: Dict[str, List[KnowledgeTriple]],
        subject_layer: int,
        object_layer: int
    ) -> Dict:
        """
        Evaluate a system with mixed training across ALL relations.
        
        Training Strategy:
        1. Train ONCE on ALL relations mixed together
        2. Evaluate ONCE on ALL test data 
        3. Analyze results by frequency buckets post-hoc
        
        Returns results broken down by cooccurrence_count buckets.
        """
        logging.info(f"🔬 Evaluating {system_name} with mixed training (layers {subject_layer}->{object_layer})")
        
        # Combine ALL training data from all relations
        all_train_triples = []
        all_test_triples = []
        
        for relation in train_data.keys():
            if relation in test_data:
                all_train_triples.extend(train_data[relation])
                all_test_triples.extend(test_data[relation])
        
        logging.info(f"   📊 Mixed training: {len(all_train_triples)} train, {len(all_test_triples)} test across {len(train_data)} relations")
        
        # Build GLOBAL vocabulary from ALL train+test triples
        all_objects = sorted({t.object for t in all_train_triples + all_test_triples})
        object_to_index = {obj: idx for idx, obj in enumerate(all_objects)}
        
        if len(object_to_index) == 0:
            logging.error("❌ Empty global vocabulary. No objects found.")
            return {'error': 'empty_vocabulary'}
        
        logging.info(f"   📚 Global vocabulary size: {len(object_to_index)} objects")
        
        try:
            # Initialize system with GLOBAL vocab size
            system = self.initialize_system(system_name, vocab_size=len(object_to_index))
            
            # Ensure system is properly moved to device
            system.to(system.probe_device)
            logging.info(f"   🔧 System initialized on device: {system.probe_device}")
            
            # Extract training activations from ALL relations (few-shot)
            logging.info("   🔄 Extracting training activations from all relations...")
            train_activations = system.extract_activations(
                all_train_triples, subject_layer, object_layer, use_few_shot=True,
                batch_size=getattr(self, 'prompt_batch_size', 16)
            )
            train_labels = self._create_labels(all_train_triples, object_to_index, system)
            # Ensure train_labels are on the correct device
            train_labels = train_labels.to(system.probe_device)
            
            # Ensure all training data is on the correct device
            train_activations = train_activations.to(system.probe_device)
            train_labels = train_labels.to(system.probe_device)
            logging.info(f"   🔧 Training data moved to device: {system.probe_device}")
            
            # Train the system ONCE on ALL mixed data
            logging.info("   🚀 Training system on ALL relations mixed...")
            train_metrics = system.train_system(
                all_train_triples, train_activations, train_labels,
                subject_layer=subject_layer, object_layer=object_layer,
                object_to_index=object_to_index,
                batch_size=getattr(self, 'train_batch_size', 4096),  # Use configurable batch size
                n_folds=self.cv_folds  # Pass the cv_folds parameter
            )
            
            # Extract test activations from ALL relations (zero-shot)
            logging.info("   🔄 Extracting test activations from all relations...")
            test_activations = system.extract_activations(
                all_test_triples, subject_layer, object_layer, use_few_shot=False,
                batch_size=getattr(self, 'prompt_batch_size', 16)
            )
            test_labels = self._create_labels(all_test_triples, object_to_index, system)
            # Ensure all test data is on the correct device
            test_activations = test_activations.to(system.probe_device)
            test_labels = test_labels.to(system.probe_device)
            logging.info(f"   🔧 Test data moved to device: {system.probe_device}")
            
            # Evaluate ONCE on ALL test data
            logging.info("   🔍 Evaluating on all test data...")
            overall_results = system.evaluate_system(
                all_test_triples, test_activations, test_labels, subject_layer, object_layer
            )
            
            # Now analyze results by frequency buckets POST-HOC
            logging.info("   📊 Analyzing results by frequency buckets...")
            cooccurrence_buckets = self._categorize_by_cooccurrence(all_test_triples)
            
            results = {}
            
            # For each frequency bucket, extract the corresponding predictions and analyze
            for bucket_name, bucket_triples in cooccurrence_buckets.items():
                if bucket_triples:
                    logging.info(f"     🔍 Analyzing bucket '{bucket_name}' ({len(bucket_triples)} samples)")
                    
                    # Find indices of bucket triples in the full test set
                    bucket_indices = []
                    for i, test_triple in enumerate(all_test_triples):
                        if test_triple in bucket_triples:
                            bucket_indices.append(i)
                    
                    if bucket_indices:
                        # Create bucket test labels on the same device as overall results
                        bucket_test_labels = self._create_labels(bucket_triples, object_to_index, system)
                        # Ensure bucket test labels are on the same device as predictions
                        bucket_test_labels = bucket_test_labels.to(overall_results.internal_predictions.device)
                        
                        # Extract bucket-specific results from overall predictions
                        bucket_results = self._extract_bucket_metrics(
                            overall_results, bucket_indices, len(bucket_triples), bucket_test_labels
                        )
                        
                        results[bucket_name] = {
                            'test_size': len(bucket_triples),
                            'metrics': bucket_results
                        }
                        logging.info(f"     ✅ Bucket '{bucket_name}' analysis complete")
                    else:
                        results[bucket_name] = {'test_size': len(bucket_triples), 'metrics': None}
                else:
                    results[bucket_name] = {'test_size': 0, 'metrics': None}
            
            results['train_metrics'] = train_metrics
            results['total_test_size'] = len(all_test_triples)
            results['global_vocab_size'] = len(object_to_index)
            
        except Exception as e:
            logging.error(f"❌ Error in mixed training evaluation: {e}")
            results = {'error': str(e)}
        
        return results
    
    def _extract_bucket_metrics(
        self,
        overall_results: 'ProbeResults',
        bucket_indices: List[int],
        bucket_size: int,
        bucket_test_labels: torch.Tensor
    ) -> 'ProbeResults':
        """
        Extract metrics for a specific frequency bucket from overall results.
        
        This creates a new ProbeResults object with metrics calculated
        only on the subset of predictions corresponding to the bucket.
        """
        from src_probe.base_probe import ProbeResults
        import torch
        
        # Extract bucket-specific predictions using indices
        # Ensure bucket_indices are on the same device as predictions
        bucket_indices_tensor = torch.tensor(bucket_indices, device=overall_results.internal_predictions.device)
        
        bucket_internal_scores = overall_results.internal_predictions[bucket_indices_tensor]
        bucket_external_scores = overall_results.external_predictions[bucket_indices_tensor]
        bucket_error_probs = overall_results.error_predictions[bucket_indices_tensor]
        
        # Recalculate accuracy@k for bucket subset
        bucket_internal_acc = {}
        bucket_external_acc = {}
        
        for k in [1, 5, 10]:  # Standard k values
            # Internal accuracy@k for bucket
            _, internal_topk = torch.topk(bucket_internal_scores, min(k, bucket_internal_scores.size(1)), dim=1)
            internal_correct = 0
            for i in range(bucket_size):
                target_indices = torch.nonzero(bucket_test_labels[i]).squeeze()
                if target_indices.numel() > 0:
                    if target_indices.numel() == 1:
                        target_indices = target_indices.unsqueeze(0)
                    if torch.any(torch.isin(internal_topk[i], target_indices)):
                        internal_correct += 1
            bucket_internal_acc[k] = internal_correct / bucket_size
            
            # External accuracy@k for bucket
            _, external_topk = torch.topk(bucket_external_scores, min(k, bucket_external_scores.size(1)), dim=1)
            external_correct = 0
            for i in range(bucket_size):
                target_indices = torch.nonzero(bucket_test_labels[i]).squeeze()
                if target_indices.numel() > 0:
                    if target_indices.numel() == 1:
                        target_indices = target_indices.unsqueeze(0)
                    if torch.any(torch.isin(external_topk[i], target_indices)):
                        external_correct += 1
            bucket_external_acc[k] = external_correct / bucket_size
        
        # Recalculate error detection metrics for bucket
        # Create true error labels for bucket on the same device
        true_errors_bucket = torch.zeros(bucket_size, device=bucket_test_labels.device)
        for i in range(bucket_size):
            target_indices = torch.nonzero(bucket_test_labels[i]).squeeze()
            if target_indices.numel() > 0:
                if target_indices.numel() == 1:
                    target_indices = target_indices.unsqueeze(0)
                # Error = external probe fails to get ground truth in top-10
                _, external_top10 = torch.topk(bucket_external_scores[i:i+1], min(10, bucket_external_scores.size(1)), dim=1)
                external_correct = torch.any(torch.isin(external_top10[0], target_indices))
                true_errors_bucket[i] = 1.0 - external_correct.float()
        
        # Convert error probabilities to predictions (ensure same device)
        if bucket_error_probs.dim() == 2 and bucket_error_probs.size(1) == 2:
            error_class_probs = torch.softmax(bucket_error_probs, dim=1)
            predicted_errors_bucket = (error_class_probs[:, 1] > 0.5).float().to(bucket_test_labels.device)
        else:
            if bucket_error_probs.dim() == 2 and bucket_error_probs.size(1) == 1:
                bucket_error_probs = bucket_error_probs.squeeze(1)
            predicted_errors_bucket = (torch.sigmoid(bucket_error_probs) > 0.5).float().to(bucket_test_labels.device)
        
        # Calculate error detection metrics for bucket
        bucket_error_accuracy = (true_errors_bucket == predicted_errors_bucket).float().mean().item()
        
        tp = (true_errors_bucket * predicted_errors_bucket).sum().item()
        fp = ((1 - true_errors_bucket) * predicted_errors_bucket).sum().item()
        fn = (true_errors_bucket * (1 - predicted_errors_bucket)).sum().item()
        
        bucket_error_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        bucket_error_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bucket_error_f1 = 2 * bucket_error_precision * bucket_error_recall / (bucket_error_precision + bucket_error_recall) if (bucket_error_precision + bucket_error_recall) > 0 else 0.0
        
        # Create bucket-specific ProbeResults (ensure all required fields are provided)
        bucket_results = ProbeResults(
            internal_accuracy_at_k=bucket_internal_acc,
            external_accuracy_at_k=bucket_external_acc,
            error_detection_accuracy=bucket_error_accuracy,
            error_detection_precision=bucket_error_precision,
            error_detection_recall=bucket_error_recall,
            error_detection_f1=bucket_error_f1,
            knowledge_gap=bucket_internal_acc.get(1, 0) - bucket_external_acc.get(1, 0),
            layer_info={'subject_layer': overall_results.layer_info['subject_layer'], 
                       'object_layer': overall_results.layer_info['object_layer']},
            internal_predictions=bucket_internal_scores,
            external_predictions=bucket_external_scores,
            error_predictions=bucket_error_probs
        )
        
        return bucket_results
    
    def _evaluate_subset(
        self,
        system,
        test_triples: List[KnowledgeTriple],
        object_to_index: Dict[str, int],
        subject_layer: int,
        object_layer: int
    ) -> ProbeResults:
        """Evaluate system on a subset of test data."""
        # Extract test activations (zero-shot)
        test_activations = system.extract_activations(
            test_triples, subject_layer, object_layer, use_few_shot=False,
            batch_size=getattr(self, 'prompt_batch_size', 16)
        )
        test_labels = self._create_labels(test_triples, object_to_index, system)
        
        # Evaluate system
        results = system.evaluate_system(
            test_triples, test_activations, test_labels, subject_layer, object_layer
        )
        
        return results
    
    def _create_labels(
        self,
        triples: List[KnowledgeTriple],
        object_to_index: Dict[str, int],
        system=None
    ) -> torch.Tensor:
        """Create multi-label targets."""
        batch_size = len(triples)
        num_classes = len(object_to_index)
        
        # Create labels on the probe device to ensure compatibility
        if system is not None and hasattr(system, 'probe_device'):
            device = system.probe_device
            # Make sure device is properly formatted as string 'cuda:0', not integer
            if isinstance(device, int):
                device = f'cuda:{device}'
        else:
            # Fallback: force to cuda:0 for consistency
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        labels = torch.zeros(batch_size, num_classes, device=device)
        for i, triple in enumerate(triples):
            if triple.object in object_to_index:
                obj_idx = object_to_index[triple.object]
                labels[i, obj_idx] = 1.0
        
        return labels
    
    def generate_comprehensive_table(
        self,
        results: Dict,
        output_dir: str,
        file_stem: Optional[str] = None
    ):
        """Generate aggregated table format - breakdown by cooccurrence_count buckets."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create table data
        table_data = []
        
        # Define cooccurrence buckets
        buckets = ["≤ 10", "(10, 100]", "(100, 1000]", "> 1000"]
        
        for probe_name, probe_results in results.items():
            model_config = self.model_configs.get(self.model_name, {})
            
            # Get layer info from first successful relation
            subject_layer, object_layer = self._get_layer_info(probe_results)
            
            # Base row data
            base_row = {
                'Probe': probe_name.upper(),
                'Model': model_config.get('display_name', self.model_name),
                'Base Model': model_config.get('base_model', 'unknown'),
                'Domain': 'General',
                'Subject Layer': subject_layer,
                'Object Layer': object_layer,
            }
            
            # Generate row for each cooccurrence bucket
            for bucket in buckets:
                bucket_metrics = self._aggregate_metrics(probe_results, bucket)
                
                if bucket_metrics:
                    bucket_row = base_row.copy()
                    bucket_row['Test Subset'] = f'Cooccur {bucket}'
                    bucket_row['Test Size (n)'] = bucket_metrics['total_test_size']
                    
                    # Internal metrics (Concept Matching - Ground Truth)
                    bucket_row['Accuracy@1'] = bucket_metrics['internal_accuracy_at_k'].get(1, 0.0)
                    bucket_row['Accuracy@5'] = bucket_metrics['internal_accuracy_at_k'].get(5, 0.0)
                    bucket_row['Accuracy@10'] = bucket_metrics['internal_accuracy_at_k'].get(10, 0.0)
                    # Removed Accuracy@25 to match paper (arXiv:2507.12379)
                    
                    # External metrics (Concept Matching - Predicted Output)
                    bucket_row['External_Accuracy@1'] = bucket_metrics['external_accuracy_at_k'].get(1, 0.0)
                    bucket_row['External_Accuracy@5'] = bucket_metrics['external_accuracy_at_k'].get(5, 0.0)
                    bucket_row['External_Accuracy@10'] = bucket_metrics['external_accuracy_at_k'].get(10, 0.0)
                    # Removed External_Accuracy@25 to match paper (arXiv:2507.12379)
                    
                    # Error detection metrics
                    bucket_row['Error Detector Accuracy'] = bucket_metrics['error_detection_accuracy']
                    bucket_row['Error Detector Precision'] = bucket_metrics.get('error_detection_precision', 0.0)
                    bucket_row['Error Detector Recall'] = bucket_metrics.get('error_detection_recall', 0.0)
                    bucket_row['Error Detector F1'] = bucket_metrics.get('error_detection_f1', 0.0)
                    
                    table_data.append(bucket_row)
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Reorder columns to match requested format (no Relation column)
        # Removed Accuracy@25 columns to match paper (arXiv:2507.12379)
        column_order = [
            'Probe', 'Model', 'Base Model', 'Domain', 'Test Subset', 'Test Size (n)',
            'Subject Layer', 'Object Layer',
            'Accuracy@1', 'Accuracy@5', 'Accuracy@10',  # Internal (Ground-Truth)
            'External_Accuracy@1', 'External_Accuracy@5', 'External_Accuracy@10',  # External (Predicted-Output)
            'Error Detector Accuracy', 'Error Detector Precision', 'Error Detector Recall', 'Error Detector F1'
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = ''
        
        df = df[column_order]
        
        # Save as CSV and formatted text with informative filename
        if file_stem is None:
            csv_filename = "comprehensive_results.csv"
            txt_filename = "comprehensive_results.txt"
        else:
            csv_filename = f"comprehensive_results_{file_stem}.csv"
            txt_filename = f"comprehensive_results_{file_stem}.txt"

        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
        # Save formatted table
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write("🚀 STATE-OF-THE-ART KNOWLEDGE TRIPLE PROBING RESULTS\n")
            f.write("=" * 120 + "\n\n")
            f.write("📊 AGGREGATED EVALUATION TABLE (Cooccurrence Count Buckets)\n\n")
            f.write("Test Size (n) Buckets:\n")
            f.write("  ≤ 10: Very rare knowledge (10 or fewer cooccurrences)\n")
            f.write("  (10, 100]: Uncommon knowledge (11-100 cooccurrences)\n")
            f.write("  (100, 1000]: Common knowledge (101-1000 cooccurrences)\n")
            f.write("  > 1000: Very common knowledge (1000+ cooccurrences)\n\n")
            f.write("Concept Matching (Internal, Ground-Truth) = Accuracy@1,5,10\n")
            f.write("Concept Matching (External, Predicted-Output) = External_Accuracy@1,5,10\n")
            f.write("Error Detection uses top-10 accuracy (matches paper arXiv:2507.12379)\n\n")
            
            # Create formatted table
            f.write(df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n" + "=" * 120 + "\n")
        
        logging.info(f"✅ Aggregated table saved to {csv_path} and {txt_path}")
        
        return df
    
    def _aggregate_metrics(self, probe_results: Dict, subset_type: str) -> Optional[Dict]:
        """Aggregate metrics across all relations for a given subset type (frequent/longtail).
        Supports both per-relation results and mixed-training single result structure.
        Computes MACRO (unweighted) averages across groups.
        """
        # Fast-path: mixed-training result structure (bucket stored at top-level)
        if isinstance(probe_results, dict) and subset_type in probe_results:
            bucket_entry = probe_results.get(subset_type, {})
            metrics = bucket_entry.get('metrics')
            test_size = bucket_entry.get('test_size', 0)
            if metrics is None or test_size == 0:
                return None
            return {
                'internal_accuracy_at_k': getattr(metrics, 'internal_accuracy_at_k', {}),
                'external_accuracy_at_k': getattr(metrics, 'external_accuracy_at_k', {}),
                'error_detection_accuracy': getattr(metrics, 'error_detection_accuracy', 0.0),
                'error_detection_precision': getattr(metrics, 'error_detection_precision', 0.0),
                'error_detection_recall': getattr(metrics, 'error_detection_recall', 0.0),
                'error_detection_f1': getattr(metrics, 'error_detection_f1', 0.0),
                'total_test_size': test_size,
            }

        # Legacy path: per-relation structure (macro/unweighted averaging)
        items: List[Dict] = []
        total_test_size = 0
        
        for relation, relation_results in probe_results.items():
            if isinstance(relation_results, dict) and 'error' in relation_results:
                continue
            if (isinstance(relation_results, dict) and subset_type in relation_results and 
                relation_results[subset_type].get('metrics') is not None and
                relation_results[subset_type].get('test_size', 0) > 0):
                metrics = relation_results[subset_type]['metrics']
                test_size = relation_results[subset_type]['test_size']
                if hasattr(metrics, 'internal_accuracy_at_k'):
                    items.append({'metrics': metrics, 'test_size': test_size})
                    total_test_size += test_size
        
        if not items:
            logging.warning(f"No valid metrics found for subset_type '{subset_type}' across all relations")
            return None
        
        def macro_mean(values: List[float]) -> float:
            vals = [v for v in values if v is not None]
            return (sum(vals) / len(vals)) if vals else 0.0
        
        # Internal and external accuracy@k (macro-average across available relations)
        internal_acc = {}
        external_acc = {}
        for k in [1, 5, 10]:
            internal_vals = []
            external_vals = []
            for item in items:
                m = item['metrics']
                if hasattr(m, 'internal_accuracy_at_k'):
                    internal_vals.append(m.internal_accuracy_at_k.get(k, 0.0))
                if hasattr(m, 'external_accuracy_at_k'):
                    external_vals.append(m.external_accuracy_at_k.get(k, 0.0))
            internal_acc[k] = macro_mean(internal_vals)
            external_acc[k] = macro_mean(external_vals)
        
        # Error detection metrics (macro-average)
        error_det_acc = macro_mean([getattr(it['metrics'], 'error_detection_accuracy', None) for it in items])
        error_prec    = macro_mean([getattr(it['metrics'], 'error_detection_precision', None) for it in items])
        error_recall  = macro_mean([getattr(it['metrics'], 'error_detection_recall', None) for it in items])
        error_f1      = macro_mean([getattr(it['metrics'], 'error_detection_f1', None) for it in items])
        
        return {
            'internal_accuracy_at_k': internal_acc,
            'external_accuracy_at_k': external_acc,
            'error_detection_accuracy': error_det_acc,
            'error_detection_precision': error_prec,
            'error_detection_recall': error_recall,
            'error_detection_f1': error_f1,
            'total_test_size': total_test_size,
        }
    
    def _get_layer_info(self, probe_results: Dict) -> Tuple[int, int]:
        """Get layer information.
        Supports both mixed-training (top-level) and per-relation structures.
        """
        # Mixed-training: layers stored at top-level
        if isinstance(probe_results, dict):
            if 'subject_layer' in probe_results and 'object_layer' in probe_results:
                return probe_results['subject_layer'], probe_results['object_layer']
        
        # Legacy per-relation structure
        if isinstance(probe_results, dict):
            for _, relation_results in probe_results.items():
                if isinstance(relation_results, dict):
                    if 'error' in relation_results:
                        continue
                    if 'subject_layer' in relation_results and 'object_layer' in relation_results:
                        return relation_results['subject_layer'], relation_results['object_layer']
        
        return 0, 0  # Fallback
    
    def _categorize_by_cooccurrence(self, triples: List[KnowledgeTriple]) -> Dict[str, List[KnowledgeTriple]]:
        """
        Categorize triples by cooccurrence_count into buckets:
        - <= 10
        - (10, 100]
        - (100, 1000]
        - > 1000
        """
        buckets = {
            "≤ 10": [],
            "(10, 100]": [],
            "(100, 1000]": [],
            "> 1000": []
        }
        
        for triple in triples:
            count = triple.cooccurrence_count
            if count <= 10:
                buckets["≤ 10"].append(triple)
            elif count <= 100:
                buckets["(10, 100]"].append(triple)
            elif count <= 1000:
                buckets["(100, 1000]"].append(triple)
            else:
                buckets["> 1000"].append(triple)
        
        return buckets


def parse_layer_list(layer_str: str) -> List[int]:
    """Parse comma-separated layer string."""
    return [int(x.strip()) for x in layer_str.split(',')]


def main():
    """Main entry point for state-of-the-art evaluation."""
    parser = argparse.ArgumentParser(
        description="State-of-the-Art Four Probing Systems Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_new.py --system lrc --model google/gemma-3-4b-pt --subject-layers 8 --object-layers 16
  python main_new.py --system all --model meta-llama/Llama-3.1-8B --subject-layers 17 --object-layers 19
  python main_new.py --system circular --model TsinghuaC3I/Llama-3.1-8B-UltraMedical --subject-layers 15 --object-layers 17

Target Models:
  - google/gemma-2-9b (Gemma 2 9B)
  - OpenMeditron/Meditron3-Gemma2-9B (Meditron3 Gemma2 9B)
  - google/gemma-3-4b-pt (Gemma 3B Base)
  - google/medgemma-4b-pt (MedGemma 4B Pretrained)  
  - google/medgemma-4b-it (MedGemma 4B Instruction-tuned)
  - meta-llama/Llama-3.1-8B (Llama 3.1 8B Base)
  - meta-llama/Llama-3.1-8B-Instruct (Llama 3.1 8B Instruct)
  - TsinghuaC3I/Llama-3.1-8B-UltraMedical (Llama 3.1 8B Medical)
  - meta-llama/Meta-Llama-3-8B (Meta Llama 3 8B)
  - Henrychur/MMed-Llama-3-8B (MMed Llama 3 8B)
  - mistralai/Mistral-7B-Instruct-v0.1 (Mistral 7B Instruct v0.1)
  - BioMistral/BioMistral-7B (BioMistral 7B)

Probing Systems:
  - lrc: Linear Relational Concepts
  - circular: Circular Probe with Joint Error Detection
  - mlp: MLP Probe with Single MLP Classifier
  - logistic: Logistic Regression with Separate Logistic Probes
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--system', 
        type=str, 
        required=True,
        choices=['lrc', 'circular', 'mlp', 'logistic', 'all'],
        help='Probing system to evaluate'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=[
            'google/gemma-2-9b', 'OpenMeditron/Meditron3-Gemma2-9B',
            'google/gemma-3-4b-pt', 'google/medgemma-4b-pt', 'google/medgemma-4b-it',
            'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B-Instruct', 'TsinghuaC3I/Llama-3.1-8B-UltraMedical',
            'meta-llama/Meta-Llama-3-8B', 'Henrychur/MMed-Llama-3-8B',
            'mistralai/Mistral-7B-Instruct-v0.1', 'BioMistral/BioMistral-7B'
        ],
        help='Model to evaluate'
    )
    
    parser.add_argument(
        '--subject-layers',
        type=str,
        required=True,
        help='Comma-separated list of subject layers (must be < object layers)'
    )
    
    parser.add_argument(
        '--object-layers', 
        type=str,
        required=True,
        help='Comma-separated list of object layers (must be > subject layers)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda, cpu, auto)'
    )
    
    parser.add_argument(
        '--prompt-batch-size',
        type=int,
        default=64,  # Increased from 16 for better GPU utilization during activation extraction
        help='Batch size for prompt/model forward during activation extraction'
    )
    
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=4096,  # Increased from 2048 for maximum GPU utilization during training
        help='Batch size for probe training (MLP/logistic)'
    )
    
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (effective batch size = batch_size * accumulation_steps)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='sota_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Maximum samples per relation (omit to use the full dataset)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds for all probing systems'
    )
    
    args = parser.parse_args()
    
    # Parse layers
    subject_layers = parse_layer_list(args.subject_layers)
    object_layers = parse_layer_list(args.object_layers)
    
    # Validate layer combinations
    for subject_layer in subject_layers:
        for object_layer in object_layers:
            if subject_layer >= object_layer:
                print(f"❌ ERROR: Subject layer ({subject_layer}) must be < object layer ({object_layer})")
                print("📋 This follows the linear-relational library requirement")
                sys.exit(1)
    
    # Determine systems to evaluate
    if args.system == 'all':
        system_names = ['lrc', 'circular', 'mlp', 'logistic']
    else:
        system_names = [args.system]
    
    print("🚀 STATE-OF-THE-ART KNOWLEDGE TRIPLE PROBING EVALUATION")
    print("=" * 60)
    print(f"🎯 Systems: {system_names}")
    print(f"🤖 Model: {args.model}")
    print(f"🔍 Subject layers: {subject_layers}")
    print(f"🔍 Object layers: {object_layers}")
    print(f"🔄 CV Folds: {args.cv_folds}")
    print(f"💾 Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = StateOfTheArtEvaluator(
            model_name=args.model,
            device=args.device,
            random_seed=args.seed,
            cv_folds=args.cv_folds
        )
        # Configure batch sizes for maximum GPU utilization
        evaluator.prompt_batch_size = args.prompt_batch_size
        evaluator.train_batch_size = args.train_batch_size
        
        # Load model
        evaluator.load_model_and_tokenizer()
        
        # Load data (relation-scoped vocab will be built per relation later)
        train_data, test_data = evaluator.load_knowledge_data(args.samples)
        
        # Main evaluation loop - MIXED TRAINING APPROACH
        all_results = {}
        
        for system_name in system_names:
            print(f"\n🔬 Evaluating {system_name.upper()} System")
            print("-" * 40)
            
            # Test each layer combination with mixed training
            best_results = None
            best_score = 0
            
            for subject_layer in subject_layers:
                for object_layer in object_layers:
                    print(f"   🔍 Testing layers {subject_layer} -> {object_layer}")
                    
                    # Evaluate this configuration with mixed training
                    results = evaluator.evaluate_system_mixed_training(
                        system_name, train_data, test_data, subject_layer, object_layer
                    )
                    
                    # Score based on frequency bucket performance
                    buckets = ["≤ 10", "(10, 100]", "(100, 1000]", "> 1000"]
                    bucket_scores = []
                    
                    for bucket in buckets:
                        if bucket in results and results[bucket]['metrics']:
                            bucket_metrics = results[bucket]['metrics']
                            if hasattr(bucket_metrics, 'internal_accuracy_at_k'):
                                internal_acc1 = bucket_metrics.internal_accuracy_at_k.get(1, 0)
                                error_acc = bucket_metrics.error_detection_accuracy
                                # Weight score by bucket preference
                                bucket_weight = {"≤ 10": 0.5, "(10, 100]": 0.7, "(100, 1000]": 1.0, "> 1000": 1.2}[bucket]
                                score = (internal_acc1 + error_acc) * bucket_weight
                                bucket_scores.append(score)
                                logging.info(f"     📈 {bucket} score: (acc@1={internal_acc1:.3f} + err={error_acc:.3f}) * {bucket_weight} = {score:.3f}")
                    
                    # Overall score is average of bucket scores
                    overall_score = sum(bucket_scores) / len(bucket_scores) if bucket_scores else 0
                    
                    if overall_score > best_score:
                        best_score = overall_score
                        best_results = results.copy()
                        best_results['subject_layer'] = subject_layer
                        best_results['object_layer'] = object_layer
                        best_results['overall_score'] = overall_score
                    
                    print(f"     📊 Overall score: {overall_score:.3f}")
            
            if best_results:
                all_results[system_name] = best_results
                print(f"   ✅ Best: S{best_results['subject_layer']}→O{best_results['object_layer']} (score: {best_score:.3f})")
            else:
                print(f"   ❌ No valid results for {system_name}")
                all_results[system_name] = {}
        
        # Generate comprehensive table
        print(f"\n📊 Generating comprehensive results table...")
        # Build informative file stem: model + systems + subject layers
        def sanitize(s: str) -> str:
            return s.replace('/', '-').replace(':', '-')

        systems_stem = '-'.join([s.upper() for s in system_names])
        layers_stem = 'S' + '-'.join(map(str, subject_layers))
        model_stem = sanitize(args.model)
        file_stem = f"{systems_stem}__{model_stem}__{layers_stem}"

        evaluator.generate_comprehensive_table(all_results, args.output_dir, file_stem=file_stem)
        
        print("🎉 Evaluation completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
