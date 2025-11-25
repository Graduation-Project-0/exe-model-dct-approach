"""
Test suite for verifying the implementation
Run this to ensure all components work correctly
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import os


def test_image_generation():
    print("Testing image generation utilities...")
    
    from utils.image_generation import (
        extract_bigrams,
        create_bigram_image,
        apply_2d_dct,
        resize_image
    )
    
    dummy_data = bytes(range(256)) * 100  # 25,600 bytes
    
    # Test bigram extraction
    bigram_freq = extract_bigrams(dummy_data)
    assert len(bigram_freq) == 65536, "Bigram frequency array should have 65536 elements"
    assert bigram_freq.sum() > 0, "Bigram frequencies should not all be zero"
    print("  ✓ Bigram extraction works")
    
    # Test bigram image creation
    bigram_img = create_bigram_image(bigram_freq)
    assert bigram_img.shape == (256, 256), f"Expected (256, 256), got {bigram_img.shape}"
    assert 0 <= bigram_img.min() <= bigram_img.max() <= 1, "Image values should be in [0, 1]"
    print("  ✓ Bigram image creation works")
    
    # Test DCT
    dct_img = apply_2d_dct(bigram_img)
    assert dct_img.shape == (256, 256), f"DCT should preserve shape"
    assert 0 <= dct_img.min() <= dct_img.max() <= 1, "DCT image should be normalized"
    print("  ✓ DCT transformation works")
    
    # Test resize
    small_img = np.random.rand(100, 100)
    resized = resize_image(small_img, (256, 256))
    assert resized.shape == (256, 256), "Resize should produce correct shape"
    print("  ✓ Image resizing works")
    
    print("✓ All image generation tests passed!\n")


def test_models():
    """Test CNN models."""
    print("Testing CNN models...")
    
    from models.cnn_models import C3C2D_SingleChannel, C3C2D_TwoChannel
    
    # Test single-channel model
    model1 = C3C2D_SingleChannel()
    dummy_input1 = torch.randn(4, 1, 256, 256)
    output1 = model1(dummy_input1)
    
    assert output1.shape == (4, 1), f"Expected (4, 1), got {output1.shape}"
    assert torch.all((output1 >= 0) & (output1 <= 1)), "Output should be in [0, 1] (sigmoid)"
    print("  ✓ Single-channel model works")
    
    # Test two-channel model
    model2 = C3C2D_TwoChannel()
    dummy_input2 = torch.randn(4, 2, 256, 256)
    output2 = model2(dummy_input2)
    
    assert output2.shape == (4, 1), f"Expected (4, 1), got {output2.shape}"
    assert torch.all((output2 >= 0) & (output2 <= 1)), "Output should be in [0, 1] (sigmoid)"
    print("  ✓ Two-channel model works")
    
    # Test gradient flow
    loss1 = output1.sum()
    loss1.backward()
    assert any(p.grad is not None for p in model1.parameters()), "Gradients should flow"
    print("  ✓ Gradient computation works")
    
    print("✓ All model tests passed!\n")


def test_data_loader():
    """Test data loading."""
    print("Testing data loader...")
    
    from utils.data_loader import MalwareImageDataset
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectories
        malware_dir = Path(tmpdir) / "malware"
        benign_dir = Path(tmpdir) / "benign"
        malware_dir.mkdir()
        benign_dir.mkdir()
        
        # Create dummy executable files
        for i in range(3):
            with open(malware_dir / f"malware{i}.exe", 'wb') as f:
                f.write(bytes(range(256)) * 100)
            with open(benign_dir / f"benign{i}.exe", 'wb') as f:
                f.write(bytes(range(256)) * 100)
        
        # Test bigram_dct mode
        dataset1 = MalwareImageDataset(tmpdir, mode='bigram_dct', max_samples=6)
        assert len(dataset1) == 6, f"Expected 6 samples, got {len(dataset1)}"
        
        img, label = dataset1[0]
        assert img.shape == (1, 256, 256), f"Expected (1, 256, 256), got {img.shape}"
        assert label in [0, 1], f"Label should be 0 or 1, got {label}"
        print("  ✓ Bigram-DCT dataset works")
        
        # Test two_channel mode
        dataset2 = MalwareImageDataset(tmpdir, mode='two_channel', max_samples=6)
        img, label = dataset2[0]
        assert img.shape == (2, 256, 256), f"Expected (2, 256, 256), got {img.shape}"
        print("  ✓ Two-channel dataset works")
    
    print("✓ All data loader tests passed!\n")


def test_training_utilities():
    """Test training utilities."""
    print("Testing training utilities...")
    
    from utils.training import MetricsTracker
    
    # Create dummy predictions
    tracker = MetricsTracker()
    
    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 1, 1, 1])
    y_scores = torch.tensor([0.1, 0.6, 0.8, 0.9])
    
    tracker.update(y_true, y_pred, y_scores)
    
    metrics = tracker.compute_metrics()
    
    assert 'accuracy' in metrics, "Metrics should include accuracy"
    assert 'precision' in metrics, "Metrics should include precision"
    assert 'recall' in metrics, "Metrics should include recall"
    assert 'f1' in metrics, "Metrics should include F1"
    assert 'auc' in metrics, "Metrics should include AUC"
    assert 'confusion_matrix' in metrics, "Metrics should include confusion matrix"
    
    # Check accuracy computation
    expected_acc = 0.75  # 3 correct out of 4
    assert abs(metrics['accuracy'] - expected_acc) < 0.01, "Accuracy computation error"
    
    print("  ✓ Metrics tracking works")
    print("✓ All training utility tests passed!\n")


def test_end_to_end_pipeline1():
    """Test complete Pipeline 1 workflow."""
    print("Testing Pipeline 1 end-to-end...")
    
    from utils.image_generation import create_bigram_dct_image
    from models.cnn_models import C3C2D_SingleChannel
    
    # Create a dummy executable
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as f:
        f.write(bytes(range(256)) * 1000)
        temp_file = f.name
    
    try:
        # Generate image
        img = create_bigram_dct_image(temp_file)
        assert img.shape == (256, 256), "Image shape should be (256, 256)"
        print("  ✓ Image generation works")
        
        # Prepare for model
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        
        # Run through model
        model = C3C2D_SingleChannel()
        model.eval()
        
        with torch.no_grad():
            output = model(img_tensor)
        
        assert output.shape == (1, 1), "Output shape should be (1, 1)"
        assert 0 <= output.item() <= 1, "Output should be in [0, 1]"
        print("  ✓ Model inference works")
        
        print("✓ Pipeline 1 end-to-end test passed!\n")
    
    finally:
        os.unlink(temp_file)


def test_end_to_end_pipeline2():
    """Test complete Pipeline 2 workflow."""
    print("Testing Pipeline 2 end-to-end...")
    
    from utils.image_generation import create_two_channel_image
    from models.cnn_models import C3C2D_TwoChannel
    
    # Create a dummy executable
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as f:
        f.write(bytes(range(256)) * 1000)
        temp_file = f.name
    
    try:
        # Generate image
        img = create_two_channel_image(temp_file)
        assert img.shape == (256, 256, 2), "Image shape should be (256, 256, 2)"
        print("  ✓ Two-channel image generation works")
        
        # Prepare for model (convert HWC to CHW)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        
        # Run through model
        model = C3C2D_TwoChannel()
        model.eval()
        
        with torch.no_grad():
            output = model(img_tensor)
        
        assert output.shape == (1, 1), "Output shape should be (1, 1)"
        assert 0 <= output.item() <= 1, "Output should be in [0, 1]"
        print("  ✓ Model inference works")
        
        print("✓ Pipeline 2 end-to-end test passed!\n")
    
    finally:
        os.unlink(temp_file)


def run_all_tests():
    print("Running Test Suit...")
    
    tests = [
        ("Image Generation", test_image_generation),
        ("CNN Models", test_models),
        ("Data Loader", test_data_loader),
        ("Training Utilities", test_training_utilities),
        ("Pipeline 1 End-to-End", test_end_to_end_pipeline1),
        ("Pipeline 2 End-to-End", test_end_to_end_pipeline2),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            failed += 1
    
    print("Test Sammry...")
    print(f"\tTotal tests: {len(tests)}")
    print(f"\tPassed: {passed}")
    print(f"\tFailed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} test(s) failed")
        
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
