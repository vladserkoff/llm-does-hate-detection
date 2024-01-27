"""Initializes the models for the hate detection app."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Tuple, Union

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from unsloth import FastLanguageModel

from .data import format_instruction, postproces_prediction

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerFast


def load_model(
    model_name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length: int = 1024,
    with_lora: bool = True,
    lora_r: int = 16,
    random_seed: int = 42,
) -> Tuple[FastLanguageModel, PreTrainedTokenizerFast]:
    """Load model from huggingface
    Args:
        model_name: name of the model checkpoint to load
        max_seq_length: max sequence length of the model
        with_lora: whether to use (Q)LoRA or not
        lora_r: size and alpha of the (Q)LoRA adapter

    When using unsloth for model finetuning, it makes sense to use the checkpoints provided by unsloth
    namespace, e.g. unsloth/mistral-7b-bnb-4bit, since they are saved as quantizied weights, thus taking less space.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_bos_token = False
    tokenizer.padding_side = "left"
    if with_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_r,
            r=lora_r,
            lora_dropout=0,
            bias="none",
            random_state=random_seed,
            max_seq_length=max_seq_length,
        )
    return model, tokenizer


def generate(
    prompt: str,
    model: FastLanguageModel,
    tokenizer: PreTrainedTokenizerFast,
    template: str,
    max_new_tokens: int = 20,
    **kwargs,
) -> str:
    """Generate and decode completion from the model
    Args:
        prompt: prompt to generate from
        model: Generative model
        template: template to use for generation
        tokenizer: model's text tokenizer
        max_new_tokens: max number of tokens to generate
    """
    if isinstance(prompt, str):
        prompt = [prompt]
    converted = [format_instruction(prompt=x, template=template) for x in prompt]
    inputs = {k: v.to("cuda") for k, v in tokenizer(converted, return_tensors="pt", padding=True).items()}
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id, **kwargs
    )
    decoded = tokenizer.batch_decode(
        outputs[:, -max_new_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return [x.rsplit("[/INST]")[-1].strip() for x in decoded]


@torch.no_grad()
def evaluate_model(
    generate_fn: Callable[[str], str], test_dataset: Dataset, batch_size: int = 32
) -> Dict[str, Union[float, str]]:
    """Evaluate the model on test dataset
    Args:
        generate_fn: function to generate completion from the model
        test_dataset: test dataset to evaluate on
    Returns:
        predictions: list of predictions
        labels: list of labels
    """
    predictions = []
    labels = []
    loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    for examples in tqdm(loader, desc="Evaluating"):
        prediction = generate_fn(examples["prompt"])
        predictions.extend([postproces_prediction(x) for x in prediction])
        labels.extend(examples["completion"])
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions),
    }
