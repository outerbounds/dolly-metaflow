"""
This file is a streamlit app template to ask questions to your Dolly model.

After you complete a run with:
```
python train_dolly.py run
```

You can interact with the model using:
```
streamlit run app.py
```
"""

# 3rd party dependencies
import streamlit as st
from metaflow import Flow, namespace, S3

# Python built-in dependencies
from time import time, sleep
import os

# Local dependencies for Dolly
from generate import generate_response, load_model_tokenizer_for_generate

def download_model(run, local_model_dir = '.'):
    "Download model from S3 storage for a Metaflow run to local_model_dir."
    with S3(run=run) as s3:
        for s3_path, _ in run.data.filepath_tuples:
            obj = s3.get(s3_path)
            os.rename(obj.path, os.path.join(local_model_dir, s3_path))

@st.cache_resource
def fetch_model(download_option, mf_namespace):
    """
    Return the model, tokenizer, and its name.
    Cache function result so it doesn't load on every streamlit refresh: https://docs.streamlit.io/library/advanced-features/caching
    """
    if download_option == 'Use my latest Metaflow run':
        namespace(mf_namespace)
        run = Flow('TrainDolly').latest_successful_run
        local_model_dir = run.data.local_output_dir.split('dolly-ops/')[1]
        os.makedirs(local_model_dir, exist_ok=True)
        if len(os.listdir(local_model_dir)) == 0:
            download_model(run) 
        model, tokenizer = load_model_tokenizer_for_generate(local_model_dir) 
        model_name = f"My Dolly from run {run.id}"
    elif download_option == 'Use latest databricks model from HuggingFace hub':
        model, tokenizer = load_model_tokenizer_for_generate("databricks/dolly-v1-6b") 
        model_name = "Databricks latest checkpoint for Dolly"
    return model, tokenizer, model_name

# Say hi and show some examples.
st.title("Ask your Dolly model questions!")
st.markdown("#### See the following examples for inspiration.")
for example in [
    "What parts does every good story have?",
    "Write a story about a person who cannot see the random forest because they are caught up in the decision trees.",
    "Explain what a recurrent neural network does. Are transformers better?"
]:
    st.markdown(f"* {example}")


# Default value
model_name = "No model"

# Ask the user which Metaflow namespace to use.
# Only matters if user selects `Use my latest Metaflow run`.
mf_namespace = st.text_input("Metaflow namespace:", value="user:ubuntu")

# Ask the user which Dolly version to use.
model, tokenizer, model_name = fetch_model(
    st.selectbox(
        'Which Dolly 🐑 do you want to use?',
        ('Use my latest Metaflow run', 'Use latest databricks model from HuggingFace hub')
    ),
    mf_namespace = mf_namespace
)
st.write(model_name + ' is selected.')

# Ask the user for a prompt.
instruction = st.text_input("Enter your instruction/question:")
 
# Upon user clicking `Ask Dolly`, generate a response.
# If no tokenizer & model are loaded, or the user hasn't specified an instruction, nothing happens.
# The `with st.spinner` is just to show the user that something is happening.
if st.button("Ask Dolly") and instruction and tokenizer and model:
    with st.spinner("Generating tokens..."):
        response = generate_response(instruction, model=model, tokenizer=tokenizer)
        if response:
            st.markdown(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")