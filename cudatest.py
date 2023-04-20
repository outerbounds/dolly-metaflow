import numpy as np
from numba import cuda
import sys

def push_cuda(array_size=1_500_000_000, num_iter=100, device=0):

    @cuda.jit
    def add_array(a, b, c):
        i = cuda.grid(1)
        if i < a.size:
            c[i] += a[i] + b[i]

    cuda.select_device(device)

    arr1 = np.ones(array_size, dtype='f4')
    arr2 = np.ones(array_size, dtype='f4')
    gpu_arr1 = cuda.to_device(arr1)
    gpu_arr2 = cuda.to_device(arr2)
    gpu_ret = cuda.device_array_like(arr1)

    threadsperblock = 256
    blockspergrid = (arr1.size + (threadsperblock - 1)) // threadsperblock

    for _ in range(num_iter):
        add_array[blockspergrid, threadsperblock](gpu_arr1, gpu_arr2, gpu_ret)
    cuda.synchronize()

    s = num_iter * (arr1 + arr2)
    res = gpu_ret.copy_to_host()
    assert np.allclose(s, res)

if __name__ == '__main__':
    cuda.select_device(int(sys.argv[1]))
    push_cuda(array_size=int(sys.argv[2]),
              num_iter=int(sys.argv[3]))