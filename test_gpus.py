import torch

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())

    # Test a small tensor operation on GPU
    x = torch.rand(3, 3).to("cuda")
    y = torch.rand(3, 3).to("cuda")
    z = x @ y
    print("Tensor operation successful on GPU. Result:\n", z)
else:
    print("Running on CPU only.")
