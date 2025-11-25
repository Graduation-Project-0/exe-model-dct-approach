"""
Main execution script for Malware Detection using Frequency Domain-Based Image Visualization
Implements both Pipeline 1 (Bigram-DCT) and Pipeline 2 (Ensemble) from the paper

Usage:
    # Train Pipeline 1 (Bigram-DCT single-channel)
    python main.py --pipeline 1 --data_dir ./data --epochs 50

    # Train Pipeline 2 (Two-channel ensemble)
    python main.py --pipeline 2 --data_dir ./data --epochs 50

    # Test a saved model
    python main.py --pipeline 1 --data_dir ./data --test_only --model_path ./checkpoints/pipeline1_best.pth
"""

import argparse
import os
import torch
from pathlib import Path

# Import custom modules
from utils.data_loader import create_data_loaders
from models.cnn_models import C3C2D_SingleChannel, C3C2D_TwoChannel, count_parameters
from utils.training import (
    train_model, 
    test_model, 
    plot_training_history,
    plot_roc_curve,
    plot_confusion_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Malware Detection using Frequency Domain-Based Image Visualization'
    )
    
    # Pipeline selection
    parser.add_argument(
        '--pipeline',
        type=int,
        choices=[1, 2],
        required=True,
        help='Pipeline to run: 1 (Bigram-DCT single-channel) or 2 (Two-channel ensemble)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing malware and benign subdirectories'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to use (for testing, default: all)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    # Data split arguments
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.7,
        help='Training data fraction (default: 0.7)'
    )
    
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation data fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.1,
        help='Test data fraction (default: 0.1)'
    )
    
    # Model saving/loading
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to saved model for testing or resuming training'
    )
    
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run testing (requires --model_path)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results and plots (default: ./results)'
    )
    
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Skip plotting results'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training (default: auto)'
    )
    
    return parser.parse_args()


def setup_directories(checkpoint_dir: str, output_dir: str):
    """Create necessary directories."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_device(device_arg: str) -> torch.device:
    """Get torch device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    return device


def run_pipeline_1(args):
    """Run Pipeline 1: Bigram-DCT single-channel CNN."""
    print("="*70)
    print("PIPELINE 1: BIGRAM-DCT FREQUENCY IMAGE (Single-Channel)")
    print("="*70)
    print()
    
    # Setup
    device = get_device(args.device)
    setup_directories(args.checkpoint_dir, args.output_dir)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        mode='bigram_dct',
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = C3C2D_SingleChannel()
    print(f"Model: 3C2D CNN (Single-Channel)")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Define paths
    model_save_path = os.path.join(args.checkpoint_dir, 'pipeline1_best.pth')
    
    # Load model if specified
    if args.model_path:
        print(f"\nLoading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training
    if not args.test_only:
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            save_path=model_save_path,
            patience=args.patience
        )
        
        # Plot training history
        if not args.no_plot:
            plot_path = os.path.join(args.output_dir, 'pipeline1_training_history.png')
            plot_training_history(history, save_path=plot_path)
    
    # Testing
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    metrics = test_model(model, test_loader, device)
    
    # Plot results
    if not args.no_plot:
        # ROC curve
        roc_path = os.path.join(args.output_dir, 'pipeline1_roc_curve.png')
        plot_roc_curve(metrics, save_path=roc_path)
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, 'pipeline1_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    return metrics


def run_pipeline_2(args):
    """Run Pipeline 2: Two-channel ensemble CNN (Byteplot + Bigram-DCT)."""
    print("="*70)
    print("PIPELINE 2: ENSEMBLE MODEL (Byteplot + Bigram-DCT)")
    print("="*70)
    print()
    
    # Setup
    device = get_device(args.device)
    setup_directories(args.checkpoint_dir, args.output_dir)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        mode='two_channel',
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        max_samples=args.max_samples,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = C3C2D_TwoChannel()
    print(f"Model: 3C2D CNN (Two-Channel)")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Define paths
    model_save_path = os.path.join(args.checkpoint_dir, 'pipeline2_best.pth')
    
    # Load model if specified
    if args.model_path:
        print(f"\nLoading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training
    if not args.test_only:
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            save_path=model_save_path,
            patience=args.patience
        )
        
        # Plot training history
        if not args.no_plot:
            plot_path = os.path.join(args.output_dir, 'pipeline2_training_history.png')
            plot_training_history(history, save_path=plot_path)
    
    # Testing
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    metrics = test_model(model, test_loader, device)
    
    # Plot results
    if not args.no_plot:
        # ROC curve
        roc_path = os.path.join(args.output_dir, 'pipeline2_roc_curve.png')
        plot_roc_curve(metrics, save_path=roc_path)
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, 'pipeline2_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    return metrics


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate arguments
    if args.test_only and args.model_path is None:
        raise ValueError("--test_only requires --model_path")
    
    # Print configuration
    print("\n" + "="*70)
    print("MALWARE DETECTION USING FREQUENCY DOMAIN-BASED IMAGE VISUALIZATION")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Pipeline:       {args.pipeline}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Device:         {args.device}")
    print(f"  Test only:      {args.test_only}")
    print()
    
    # Run selected pipeline
    if args.pipeline == 1:
        metrics = run_pipeline_1(args)
    elif args.pipeline == 2:
        metrics = run_pipeline_2(args)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Pipeline {args.pipeline} Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Pipeline {args.pipeline} AUC:      {metrics.get('auc', 0):.4f}")
    print("="*70)
    print("\nTraining complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Model saved to:   {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
