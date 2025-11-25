"""
Example script demonstrating usage of both pipelines
Shows how to use individual components for custom workflows
"""

import torch
import numpy as np
from pathlib import Path

# Import components
from utils.image_generation import (
    create_bigram_dct_image,
    create_byteplot_image,
    create_two_channel_image
)
from models.cnn_models import C3C2D_SingleChannel, C3C2D_TwoChannel


def example_1_generate_images():
    print("\nEXAMPLE 1: Generate Images from Executable...")
    
    # Path to an executable (replace with your file)
    exe_path = r"C:\Users\CRIZMA\Downloads\OfficeSetup.exe"
    
    if not Path(exe_path).exists():
        print(f"File not found: {exe_path}")
        print("Please provide a valid executable path.")
        return
    
    print(f"\nProcessing: {exe_path}")
    
    # Generate Bigram-DCT image (Pipeline 1)
    print("\n1. Generating Bigram-DCT image...")
    bigram_dct = create_bigram_dct_image(exe_path)
    print(f"   Shape: {bigram_dct.shape}")
    print(f"   Range: [{bigram_dct.min():.4f}, {bigram_dct.max():.4f}]")
    
    # Generate Byteplot image
    print("\n2. Generating Byteplot image...")
    byteplot = create_byteplot_image(exe_path)
    print(f"   Shape: {byteplot.shape}")
    print(f"   Range: [{byteplot.min():.4f}, {byteplot.max():.4f}]")
    
    # Generate Two-channel image (Pipeline 2)
    print("\n3. Generating Two-channel ensemble image...")
    two_channel = create_two_channel_image(exe_path)
    print(f"   Shape: {two_channel.shape}")
    print(f"   Channel 0 (Byteplot) range: [{two_channel[:,:,0].min():.4f}, {two_channel[:,:,0].max():.4f}]")
    print(f"   Channel 1 (Bigram-DCT) range: [{two_channel[:,:,1].min():.4f}, {two_channel[:,:,1].max():.4f}]")
    
    print("\n✓ Image generation complete!")


def example_2_model_inference():
    print("\nEXAMPLE 2: Model Inference")
    
    # Check for saved model
    model_path = "./checkpoints/pipeline1_best.pth"
    
    if not Path(model_path).exists():
        print(f"\nModel not found: {model_path}")
        print("Please train a model first using main.py")
        return
    
    print(f"\nLoading model from: {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = C3C2D_SingleChannel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    # Generate a sample image
    exe_path = "./data/malware/sample.exe"
    if not Path(exe_path).exists():
        print(f"\nExecutable not found: {exe_path}")
        print("Please provide a valid executable path.")
        return
    
    print(f"\nProcessing executable: {exe_path}")
    bigram_dct = create_bigram_dct_image(exe_path)
    
    # Prepare for model input
    image_tensor = torch.from_numpy(bigram_dct).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output >= 0.5).item()
        confidence = output.item()
    
    print(f"\nPrediction:")
    print(f"  Class: {'Malware' if prediction == 1 else 'Benign'}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Raw score: {output.item():.6f}")


def example_3_batch_processing():
    print("\nEXAMPLE 3: Batch Processing")
    
    # Example file list (replace with your files)
    file_list = [
        "./data/malware/sample1.exe",
        "./data/malware/sample2.exe",
        "./data/benign/sample1.exe"
    ]
    
    # Filter existing files
    existing_files = [f for f in file_list if Path(f).exists()]
    
    if not existing_files:
        print("\nNo valid executable files found.")
        print("Please provide valid paths in the file_list.")
        return
    
    print(f"\nProcessing {len(existing_files)} files...")
    
    images = []
    for i, file_path in enumerate(existing_files):
        print(f"\n{i+1}. {file_path}")
        try:
            img = create_bigram_dct_image(file_path)
            images.append(img)
            print(f"   ✓ Generated image: {img.shape}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print(f"\n✓ Successfully processed {len(images)}/{len(existing_files)} files")


def example_4_compare_pipelines():
    print("\nEXAMPLE 4: Compare Pipeline Outputs")
    
    exe_path = "./data/malware/sample.exe"
    
    if not Path(exe_path).exists():
        print(f"\nFile not found: {exe_path}")
        print("Please provide a valid executable path.")
        return
    
    print(f"\nProcessing: {exe_path}")
    
    # Pipeline 1
    print("\nPipeline 1 (Bigram-DCT):")
    img1 = create_bigram_dct_image(exe_path)
    print(f"  Output shape: {img1.shape}")
    print(f"  Channels: 1 (grayscale)")
    print(f"  Value range: [{img1.min():.4f}, {img1.max():.4f}]")
    print(f"  Non-zero pixels: {np.count_nonzero(img1)}/{img1.size}")
    
    # Pipeline 2
    print("\nPipeline 2 (Ensemble):")
    img2 = create_two_channel_image(exe_path)
    print(f"  Output shape: {img2.shape}")
    print(f"  Channels: 2 (byteplot + bigram-dct)")
    print(f"  Channel 0 range: [{img2[:,:,0].min():.4f}, {img2[:,:,0].max():.4f}]")
    print(f"  Channel 1 range: [{img2[:,:,1].min():.4f}, {img2[:,:,1].max():.4f}]")
    
    print("\nKey Differences:")
    print("  - Pipeline 1: Single frequency-domain representation")
    print("  - Pipeline 2: Combines spatial (byteplot) + frequency (DCT)")
    print("  - Pipeline 2 typically achieves higher accuracy (~96% vs ~94%)")


def example_5_model_comparison():
    print("\nEXAMPLE 5: Model Architecture Comparison")
    
    from models.cnn_models import count_parameters
    
    # Pipeline 1 model
    model1 = C3C2D_SingleChannel()
    params1 = count_parameters(model1)
    
    print("\nPipeline 1 Model (Single-Channel):")
    print(f"  Input shape: (1, 256, 256)")
    print(f"  Total parameters: {params1:,}")
    print(f"  Architecture: 3 Conv layers + 3 Dense layers")
    
    # Pipeline 2 model
    model2 = C3C2D_TwoChannel()
    params2 = count_parameters(model2)
    
    print("\nPipeline 2 Model (Two-Channel):")
    print(f"  Input shape: (2, 256, 256)")
    print(f"  Total parameters: {params2:,}")
    print(f"  Architecture: 3 Conv layers + 3 Dense layers")
    
    print(f"\nParameter difference: {params2 - params1:,}")
    print("  (Due to 2 input channels vs 1 in first conv layer)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    dummy_input1 = torch.randn(1, 1, 256, 256)
    output1 = model1(dummy_input1)
    print(f"  Pipeline 1 output shape: {output1.shape}")
    
    dummy_input2 = torch.randn(1, 2, 256, 256)
    output2 = model2(dummy_input2)
    print(f"  Pipeline 2 output shape: {output2.shape}")
    
    print("\n✓ Both models working correctly!")


def main():    
    try:
        example_1_generate_images()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_2_model_inference()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_3_batch_processing()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_4_compare_pipelines()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example_5_model_comparison()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")

if __name__ == "__main__":
    main()
