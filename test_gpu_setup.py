"""
This file gives you a reference point for analyzing GPU performance for your config in train_dolly.py.
You can run this file with various parameters to see how it affects GPU performance.
Performance is measured using the @gpu_profile decorator in my_decorators.py.

After you complete a run with:
```
python test_gpu_setup.py run
```

You can view the results with:
```
python test_gpu_setup.py card view train
```
"""

from metaflow import FlowSpec, step, Parameter, resources, parallel_map, conda
from my_decorators import gpu_profile, pip

N_CPU = 8
N_GPU = 8
MEMORY = 32000

class TestDollyGPUSetup(FlowSpec):

    # config
    TRAINING_OUTPUT_DIR = Parameter(name="training_output_dir", default="training_output", help="Directory to store training output.")
    DEEPSPEED_CONFIG = Parameter(name="config", default="ds_config.json", help="DeepSpeed config file.")

    num_iter = Parameter("num_iter", default=10, help="number of iterations per task")
    num_tasks = Parameter("num_tasks", default=1, help="number of parallel GPU tasks")

    @step
    def start(self):
        print("Running test on GPU setup. Use to match the Dolly configuration next door in train_dolly.py...")
        print("N_CPU: ", N_CPU)
        print("N_GPU: ", N_GPU)
        print("MEMORY: ", MEMORY)
        self.next(self.train)

    @resources(cpu=N_CPU, gpu=N_GPU, memory=MEMORY)
    @gpu_profile(interval=1)
    @step
    def train(self):

        import cudatest
        import time
        from random import seed, randint

        seed(self.index)
        for i in range(self.num_iter):
            params = [
                {
                    "array_size": randint(10_000_000, 10_000_000),
                    "num_iter": randint(10, 1000),
                    "device": i,
                }
                for i in range(self.gpu_profile_num_gpus)
            ]
            parallel_map(lambda x: cudatest.push_cuda(**x), params)
            print(f"iteration {i} done")

        self.next(self.end)

    @step
    def end(self):
        print(f"Test complete!")

if __name__ == '__main__':
    TestDollyGPUSetup()