"""
This flow wraps the training code for Databricks Dolly model. 
You can find the original source code here: https://github.com/databrickslabs/dolly

The extensions this flow has include:
    - Easy to package to run on AWS Batch or Kubernetes
    - GPU profiling
    - A small streamlit app to interact with the model after training in app.py
"""

from metaflow import FlowSpec, step, Parameter, resources, environment, S3
from metaflow import IncludeFile
import subprocess
from my_decorators import gpu_profile, pip
from consts import *

def print_volumes():
    print("Volumes:")
    subprocess.run(["df", "-h"])
    print("")
    print("Mounted volumes:")
    subprocess.run(["mount"])
    print("")

class TrainDolly(FlowSpec):

    # config
    training_output_dir = Parameter(name="training_output_dir", default="training_output", help="Directory to store training output.")
    deepspeed_config = IncludeFile(name="config", default="ds_config.json", help="DeepSpeed config file.")

    # hyperparameters
    learning_rate = Parameter("learning_rate", default="1e-5", help="Learning rate for training.")
    batch_size = Parameter("batch_size", default="16", help="Batch size for training & evaluation.")

    debug_log = Parameter("debug_log", is_flag=True, default=False, help="Whether to log debug messages.")
    sample_percentage = Parameter("sample-percentage", default=None, help="Percentage of data to use for training.")
    push_model_to_s3 = Parameter("push-model-to-s3", default=True, help="Whether to push model to S3 after training.")

    @step
    def start(self):
        print(self.deepspeed_config)
        self.next(self.train)

    # # (NOTE): Uncomment this or include these dependencies if using @kubernetes or @batch for compute layer.
    # @pip(libraries={                     
    #     "accelerate": "0.18.0",         
    #     "click": "8.0.3",                
    #     "datasets": "2.8.0",             
    #     "deepspeed": "0.8.3",           
    #     "transformers[torch]": "4.25.1",
    #     "watchdog": "2.1.9"              
    # })
    @resources(cpu=N_CPU, gpu=N_GPU, memory=MEMORY)
    @environment(vars = {"TOKENIZERS_PARALLELISM": "false"})
    @gpu_profile(interval=1)
    @step
    def train(self):
        import os 
        import subprocess
        from datetime import datetime
        import tempfile
        import json

        # Configure local output directory.
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        model_name = "dolly"
        checkpoint_dir_name = f"{model_name}__{timestamp}"
        root_path = os.getcwd()
        local_training_root = os.path.join(os.getcwd(), self.training_output_dir)
        os.makedirs(local_training_root, exist_ok=True)
        self.local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)

        # Configure deepspeed.
        ds_config_file = tempfile.NamedTemporaryFile(mode="w")
        ds_config_file.write(self.deepspeed_config)
        ds_config_file.seek(0)
        with open(ds_config_file.name) as f:
            print(json.dumps(json.load(f), indent=4))
        
        # This will not run trivially without volume mounts. 
        print_volumes()

        print("Calling deepspeed process...\n\n")
        subprocess.run(
            [
                "deepspeed", 
                "--num_gpus=%d" % N_GPU, 
                "--module", MODEL_TRAINING_SCRIPT, 
                "--deepspeed", ds_config_file.name,
                "--epochs", "1",
                "--local-output-dir", self.local_output_dir,
                "--per-device-train-batch-size", self.batch_size,
                "--per-device-eval-batch-size", self.batch_size,
                "--lr", self.learning_rate
            ] + (["--debug-log"] if self.debug_log else []) + (["--sample-percentage", self.sample_percentage] if self.sample_percentage else []),
            check=True
        )
        print("\nDeepspeed process completed!")

        # Put all files in the local_output_directory in S3 bucket, versioned with this flow run.
        if self.push_model_to_s3:
            print("Writing model to s3...\n\n")
            self.s3_output_dir = os.path.join(self.training_output_dir, checkpoint_dir_name)
            with S3(run=self) as s3:
                self.filepath_tuples = [
                    (f"{self.s3_output_dir}/{f}", os.path.join(self.local_output_dir, f)) 
                    for f in os.listdir(self.local_output_dir) 
                    if os.path.isfile(os.path.join(self.local_output_dir, f))
                ]
                print(s3.put_files(self.filepath_tuples))

            # This conditional block enables resuming from this checkpoint in a subsequent flow or a notebook:
                # run = Flow('TrainDolly').latest_run
                # local_output_path = run.data.local_output_path
                # S3(run=flow.run) as s3:
                    # for local_path, s3_path in run.data.filepath_tuples:
                        # obj = s3.get(s3_path)
                        # os.rename(obj.path, local_path)
                # model = transformers.AutoModelForCausalLM.from_pretrained(local_output_path, ...)

            # Why do this?
                # You can download from s3 to your favorite environment, or use the model to make an application.
                # You can resume training from this checkpoint easily, by passing flow ids. This is useful if you want to incrementally train the model.
            # Why not do this?
                # The model is big ~25GB so there is storage cost * number of times you run a flow with `self.push_model_to_s3=True`. 
                # It can take a few minutes to push.  

        self.next(self.end)

    @step
    def end(self):
        print(f"\n\ Woohoo! You've trained your own LLM! \U0001f389 \U0001f389 \U0001f389")

if __name__ == '__main__':
    TrainDolly()