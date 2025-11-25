"""
Configuration file for malware detection system
Customize training parameters, model architecture, and paths here
"""

# Data configuration
DATA_CONFIG = {
    'data_dir': './data',
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    'max_samples': None,  # None for all samples
    'num_workers': 0,     # Number of data loading workers
}

# Model configuration
MODEL_CONFIG = {
    # Image dimensions
    'image_size': 256,
    
    # CNN architecture (as per paper)
    'conv_channels': [32, 64, 128],
    'kernel_size': 3,
    'pool_size': 2,
    'fc_layers': [512, 256],
    'dropout_rate': 0.5,
    
    # Single vs Two channel
    'pipeline_1_channels': 1,  # Bigram-DCT only
    'pipeline_2_channels': 2,  # Byteplot + Bigram-DCT
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,  # Early stopping patience
    'device': 'auto',  # 'auto', 'cpu', or 'cuda'
}

# Image generation configuration
IMAGE_CONFIG = {
    # Bigram settings
    'zero_out_bigram_0000': True,  # As per paper
    'bigram_normalization': True,
    
    # DCT settings
    'dct_normalization': 'ortho',
    
    # Byteplot settings
    'byteplot_resize_method': 'bilinear',
}

# Paths configuration
PATH_CONFIG = {
    'checkpoint_dir': './checkpoints',
    'output_dir': './results',
    'log_dir': './logs',
}

# Evaluation configuration
EVAL_CONFIG = {
    'plot_training_history': True,
    'plot_roc_curve': True,
    'plot_confusion_matrix': True,
    'save_predictions': False,
    'prediction_threshold': 0.5,
}

# Logging configuration
LOG_CONFIG = {
    'log_level': 'INFO',
    'log_to_file': False,
    'log_file': './logs/training.log',
}


def get_config(pipeline: int = 1):
    """
    Get complete configuration for specified pipeline.
    
    Args:
        pipeline: 1 for Bigram-DCT, 2 for Ensemble
        
    Returns:
        Dictionary with all configuration parameters
    """
    config = {
        'pipeline': pipeline,
        'data': DATA_CONFIG.copy(),
        'model': MODEL_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'image': IMAGE_CONFIG.copy(),
        'paths': PATH_CONFIG.copy(),
        'eval': EVAL_CONFIG.copy(),
        'log': LOG_CONFIG.copy(),
    }
    
    # Set correct number of channels based on pipeline
    if pipeline == 1:
        config['model']['input_channels'] = MODEL_CONFIG['pipeline_1_channels']
    elif pipeline == 2:
        config['model']['input_channels'] = MODEL_CONFIG['pipeline_2_channels']
    
    return config


def print_config(config):
    """Print configuration in a readable format."""
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    
    for section, params in config.items():
        if isinstance(params, dict):
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {params}")
    
    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Pipeline 1 Configuration:")
    config1 = get_config(pipeline=1)
    print_config(config1)
    
    print("\n\nPipeline 2 Configuration:")
    config2 = get_config(pipeline=2)
    print_config(config2)
