"""Fine-tune LLM for hate speech detection."""

from __future__ import annotations

import os
from functools import partial

import fire
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from helpers.data import format_instruction, load_dataset
from helpers.logger import logging
from helpers.models import evaluate_model, generate, load_model
from helpers.util import ensure_reproducibility

LOG = logging.getLogger(__name__)


def main(
    model_name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    dataset_size: int = 14400,
    batch_size: int = 64,
    max_seq_length: int = 512,
    use_lora: bool = True,
    lora_r: int = 16,
    max_steps: int = -1,
    max_epochs: int = 1,
    random_seed: int = 182736827,
    export_as_gguf: bool = True,
) -> None:
    """Fine-tune LLM for hate speech detection.
    Args:
        model_name: name of the model checkpoint to use for fine-tuning
        dataset_size: total size of the dataset to load
        batch_size: batch size for training
        max_seq_length: max sequence length of the model
        use_lora: whether to use (Q)LoRA or not
        lora_r: size and alpha of the (Q)LoRA adapter
        max_steps: if provided, will only train for this number of steps ignoring the size of the dataset
        max_epochs: if max_steps is -1, will train for this number of epochs
        random_seed: random seed for reproducibility
        export_as_gguf: whether to export the model as GGUF after training
    """
    ensure_reproducibility(seed=random_seed)
    device = torch.device("cuda")
    template = open(os.path.join(os.path.dirname(__file__), "config", "TEMPLATE")).read()
    LOG.info(f"Using device {device}")
    LOG.info(f"Loading dataset with {dataset_size} examples")
    train_dataset, test_dataset = load_dataset(size=dataset_size)
    LOG.info(f"Loading model {model_name}")
    model, tokenizer = load_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        with_lora=use_lora,
        lora_r=lora_r,
        random_seed=random_seed,
    )
    LOG.info(f"Fine-tuning model {model_name}")
    training_args = TrainingArguments(
        output_dir=os.path.join("../results", model_name),
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=100,
        save_steps=100,
        save_total_limit=1,
        optim="adamw_8bit",
        seed=random_seed,
        learning_rate=1e-4,
        weight_decay=0.01,
        run_name=model_name,
        disable_tqdm=False,
    )

    def formatting_func(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for prompt, completion in zip(examples["prompt"], examples["completion"]):
                output_texts.append(
                    format_instruction(prompt=prompt, completion=completion, template=template)
                    + " "
                    + tokenizer.eos_token
                )
            return output_texts
        else:
            return (
                format_instruction(prompt=examples["prompt"], completion=examples["completion"])
                + " "
                + tokenizer.eos_token
            )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,
    )
    # Evaluate the model before training
    generate_fn = partial(generate, model=model, tokenizer=tokenizer, template=template, max_new_tokens=20)
    no_tune_eval = evaluate_model(generate_fn=generate_fn, test_dataset=test_dataset, batch_size=batch_size)
    LOG.info("Pre-training evaluation: %s", no_tune_eval["classification_report"])

    trainer.train()

    post_tune_eval = evaluate_model(generate_fn=generate_fn, test_dataset=test_dataset, batch_size=batch_size)
    LOG.info("Post-training evaluation: %s", post_tune_eval["classification_report"])

    trainer.save_model()
    if export_as_gguf:
        model.save_pretrained_gguf(os.path.join("../results/ggufed/", model_name), tokenizer)


if __name__ == "__main__":
    fire.Fire(main)
