# Original source: https://github.com/databrickslabs/dolly/blob/master/training/trainer.py

import logging
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from consts import *

logger = logging.getLogger(__name__)


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                if np.array_equal(response_token_ids, batch["labels"][i, idx : idx + len(response_token_ids)]):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                raise RuntimeError("Could not find response key token IDs")

            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(training_data_id: str = DEFAULT_TRAINING_DATASET, split: str = "train") -> Dataset:
    logger.info(f"Loading {training_data_id} dataset")
    dataset: Dataset = load_dataset(training_data_id)[split]
    logger.info("Found %d rows", dataset.num_rows)

    # Remove empty responses
    dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith("### Response:"))

    def _func(rec):
        rec["text"] += "\n\n### End"
        return rec

    dataset = dataset.map(_func)

    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH, seed=DEFAULT_SEED) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int, optional): Maximum number of tokens to emit from tokenizer. Defaults to MAX_INPUT_LENGTH.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset()

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset

def train(
    local_output_dir,
    epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    lr,
    seed,
    deepspeed,
    gradient_checkpointing,
    local_rank,
    bf16,
    test_size=1000,
    debug_log=False,
    sample_percentage=None,
):
    set_seed(seed)
    if debug_log:
        transformers.utils.logging.set_verbosity_debug()

    model, tokenizer = get_model_tokenizer(gradient_checkpointing=gradient_checkpointing)
    logger.info("Model Loaded to disk")

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, seed=seed,)
    logger.info("preprocessed the dataset ")

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed, train_size=sample_percentage)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )    

    # if not dbfs_output_dir:
    #     logger.warn("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir, # The output directory where the model predictions and checkpoints will be written.
        per_device_train_batch_size=per_device_train_batch_size, # The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size=per_device_eval_batch_size, # The batch size per GPU/TPU core/CPU for evaluation.
        fp16=not bf16, # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        bf16=bf16, # Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda)
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing, # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        logging_dir=f"{local_output_dir}/runs", # TensorBoard log directory
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100, # Number of update steps between two evaluations if evaluation_strategy="steps". Will default to the same value as logging_steps if not set.
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Done.")

@click.command()
@click.option(
    "--local-output-dir", type=str, help="Write directly to this local path", required=True
)
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
@click.option("--debug-log", type=bool, default=False, help="Whether to log debug messages.")
@click.option("--sample-percentage", type=float, default=None, help="Percentage of dataset to use for training.")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise