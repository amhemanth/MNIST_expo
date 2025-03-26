import unittest
import torch
import torch.nn as nn
import time
from model import Net
from torchvision import datasets, transforms

class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = Net()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_parameter_count(self):
        """Test that model has less than 20k parameters"""
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertLess(param_count, 20000, f"Model has {param_count} parameters, should be less than 20000")
        print(f"\nTotal parameters: {param_count}")

    def test_batch_normalization(self):
        """Test presence of batch normalization layers"""
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in self.model.modules())
        self.assertTrue(has_bn, "Model should have batch normalization layers")
        bn_count = sum(1 for m in self.model.modules() if isinstance(m, nn.BatchNorm2d))
        print(f"\nBatch normalization layers: {bn_count}")

    def test_dropout(self):
        """Test presence of dropout"""
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())
        self.assertTrue(has_dropout, "Model should have dropout layer")
        dropout_layers = [m for m in self.model.modules() if isinstance(m, nn.Dropout)]
        print(f"\nDropout probabilities: {[layer.p for layer in dropout_layers]}")

    def test_architecture(self):
        """Test presence of FC layer or Global Average Pooling"""
        has_fc = any(isinstance(m, nn.Linear) for m in self.model.modules())
        has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in self.model.modules())
        self.assertTrue(has_fc or has_gap, "Model should have either FC layer or Global Average Pooling")

    def test_forward_pass(self):
        """Test forward pass with sample input"""
        batch_size = 64
        x = torch.randn(batch_size, 1, 28, 28).to(self.device)
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size, 10), "Output shape should be [batch_size, 10]")

class TestModelPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up data loaders
        cls.batch_size = 1000
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = Net().to(cls.device)

        # Load a small subset of data for quick testing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Added download=True to ensure dataset is available
        cls.test_data = datasets.MNIST('../data', train=False, 
                                     transform=transform, 
                                     download=True)
        cls.test_loader = torch.utils.data.DataLoader(cls.test_data, batch_size=cls.batch_size)

    def test_model_output_range(self):
        """Test if model outputs valid probabilities"""
        self.model.eval()
        data, _ = next(iter(self.test_loader))
        data = data.to(self.device)
        with torch.no_grad():
            output = self.model(data)
        
        # Test if output is in valid probability range
        self.assertTrue(torch.allclose(torch.exp(output).sum(dim=1), 
                                     torch.ones(len(data)).to(self.device), 
                                     atol=1e-6))

    def test_batch_inference_speed(self):
        """Test if model can process a batch quickly"""
        self.model.eval()
        data, _ = next(iter(self.test_loader))
        data = data.to(self.device)

        # Warm-up run
        with torch.no_grad():
            _ = self.model(data)

        # Measure time using time.perf_counter for both CPU and GPU
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = self.model(data)
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # Wait for GPU to finish
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # More lenient time check for CPU
        time_limit = 1000 if self.device.type == "cpu" else 100
        self.assertLess(inference_time, time_limit, 
                       f"Inference took {inference_time:.2f}ms, should be less than {time_limit}ms on {self.device.type}")
        print(f"\nBatch inference time on {self.device.type}: {inference_time:.2f}ms")

    def test_model_stability(self):
        """Test if model gives consistent outputs"""
        self.model.eval()
        data, _ = next(iter(self.test_loader))
        data = data.to(self.device)
        
        with torch.no_grad():
            output1 = self.model(data)
            output2 = self.model(data)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

if __name__ == '__main__':
    unittest.main(verbosity=2) 