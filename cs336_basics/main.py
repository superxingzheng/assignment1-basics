"""
stanford-cs336/assignment1-basics/cs336_basics/main.py
Last updated by: Zheng Xing <superxingzheng@gmail.com>
Reference: https://stanford-cs336.github.io/spring2024/
"""

import torch


def test_gpu():
    """
    Using PyTorch's convenient functions to find all CUDA devices.
    And test all CUDA devices by allocating memory for small tensors on them.
    """
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")

    if not torch.cuda.is_available():
        print("There is no CUDA available.")
        return
    else:
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            with torch.cuda.device(i):
                print(f"cuda:{torch.cuda.current_device()}")
                properties = torch.cuda.get_device_properties(i)
                print(properties)
                z = torch.zeros(32, 32).cuda() # z is created directly on GPU 0
                assert z.device == torch.device("cuda", i)
                memory_allocated = torch.cuda.memory_allocated()
                assert memory_allocated  == 32 * 32 * 4

                y = x.to(torch.cuda.current_device())  # x is copied into a tensor in GPU 0, y
                assert y.device == torch.device("cuda", i)

                memory_allocated = torch.cuda.memory_allocated()
                assert memory_allocated == 2 * (32 * 32 * 4)  # One allocated for y and one allocated for z.

        print("CUDA passed tests.")


def main():
    test_gpu()


if __name__ == "__main__":
    main()