"""Data loding and preprocessing for hate speech LLM finetuning"""

import random
from multiprocessing import cpu_count
from typing import Any, Dict, Optional

import datasets

from .logger import logging

LOG = logging.getLogger(__name__)


def load_dataset(
    name: str = "ucberkeley-dlab/measuring-hate-speech",
    size: Optional[int] = 25600,
):
    """Load and preprocess dataset from https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

    Args:
        name: name of the dataset
        size: number of examples to load, if None load the whole dataset

    N.B. Full dataset is 135556 examples, which is quite large for a demo.

    The dataset provides a continuous measure of hate speech, to convert it to classification labels we follow the
    instructions from the dataset card: "hate_speech_score - continuous hate speech measure, where higher = more hateful
    and lower = less hateful. > 0.5 is approximately hate speech, < -1 is counter or supportive speech,
    and -1 to +0.5 is neutral or ambiguous."

    The dataset is preprocessed into instruction format for Supervised Fine-tuning with TRL library.
    """
    dataset = datasets.load_dataset(name, split="train")
    if size:
        indices = random.sample(range(len(dataset)), size)
        dataset = dataset.select(indices)
    dataset = dataset.map(
        preprocess_example,
        num_proc=cpu_count(),
        load_from_cache_file=True,
    )
    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in ["prompt", "completion"]])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    LOG.info(f"Loaded dataset {name} with {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    return train_dataset, test_dataset


def preprocess_example(
    example: Dict[str, Any],
) -> Dict[str, str]:
    """Preprocess example from hate speech dataset"""
    prompt = example["text"]
    label = hate_speech_score_to_label(example["hate_speech_score"])
    return {"prompt": prompt, "completion": label}


def hate_speech_score_to_label(hate_speech_score: float) -> str:
    """The dataset provides a continuous measure of hate speech, to convert it to classification labels we follow the
    instructions from the dataset card: "hate_speech_score - continuous hate speech measure, where higher = more hateful
    and lower = less hateful. > 0.5 is approximately hate speech, < -1 is counter or supportive speech,
    and -1 to +0.5 is neutral or ambiguous."""
    return "Supportive" if hate_speech_score < -1 else ("Hate" if hate_speech_score > 0.5 else "Neutral")


def format_instruction(
    prompt: str,
    template: str,
    completion: str = "",
) -> str:
    """Format instruction for supervised finetuning"""
    prompt = template.replace("{{ .Prompt }}", prompt)  # the template is in OLLAMA format, a bit weird
    if completion:
        prompt += completion
    return prompt


def postproces_prediction(prediction: str) -> str:
    """Standardize prediction to match dataset labels"""
    if "supportive" in prediction.lower():
        return "Supportive"
    elif "hate" in prediction.lower():
        return "Hate"
    elif "neutral" in prediction.lower():
        return "Neutral"
    else:
        return "Unknown"
