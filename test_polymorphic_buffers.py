import torch
from vllm.v1.worker.gpu.buffer_utils import DeviceMemoryManager

def test():
    device = torch.device("cpu")
    manager = DeviceMemoryManager(device)
    writer = manager.get_staged_writer(10, torch.int32)
    
    print(f"Writer type: {type(writer)}")
    # Check property rename
    tensor = writer.device_tensor
    print(f"Tensor device: {tensor.device}")
    
    # Check stage_write with empty list (should not crash)
    writer.stage_write(0, 0, [])
    writer.apply_write()
    print("Empty stage_write passed")
    
    # Check actual write
    writer.stage_write(0, 0, [42])
    writer.apply_write()
    assert writer.device_tensor[0] == 42
    print("Actual write passed")

if __name__ == "__main__":
    test()
