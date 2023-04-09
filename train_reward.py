# coding:UTF-8

from typing import Any, List, Union
import evaluate
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer
)
from transformers.utils import PaddingStrategy
from dataclasses import dataclass, field
from typing import Optional, Dict
from peft import LoraConfig, get_peft_model


IGNORE_INDEX = -100

accuracy = evaluate.load("accuracy")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bigscience/bloomz-1b1")


@dataclass
class DataArguments:
    data_path: str = field(default='data/reward_train.json', metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default="cache/")
    optim: str = field(default="adafactor")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: Optional[str] = field(default="model/reward_model/")
    remove_unused_columns:  Optional[bool] = field(default=False)
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    """
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    """
    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})


@dataclass
class DataFormat(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples):
        new_examples = {"text_j": [], "text_k": []}
        print(examples)
        for prompt, good, bad in zip(examples["prompt"], examples["good"], examples["bad"]):
            new_examples["text_j"].append(
                prompt + " " + self.tokenizer.bos_token + " " + good
            )
            new_examples["text_k"].append(
                prompt + " " + self.tokenizer.bos_token + " " + bad
            )

        return new_examples


@dataclass
class PreprocessFunction(object):

    tokenizer: PreTrainedTokenizer

    def __call__(self, examples):
        tokenized_j = self.tokenizer(examples["text_j"], truncation=True)
        tokenized_k = self.tokenizer(examples["text_k"], truncation=True)
        return {
            "input_ids_j": tokenized_j["input_ids"],
            "attention_mask_j": tokenized_j["attention_mask"],
            "input_ids_k": tokenized_k["input_ids"],
            "attention_mask_k": tokenized_k["attention_mask"],
        }


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, problem_type="regression")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    #model = get_peft_model(model, lora_config)

    # Load the human comparisons dataset for tuning the reward model.
    ds = load_dataset('json', data_files=data_args.data_path)  # , split='train') #, streaming=True)
    num_proc = 8  # Can adjust to be higher if you have more processors
    original_columns = ds["train"].column_names
    ds = ds.map(DataFormat(tokenizer),
                batched=True, num_proc=num_proc, remove_columns=original_columns)
    tokenized_ds = ds.map(PreprocessFunction(tokenizer),
                          batched=True, num_proc=num_proc, remove_columns=["text_j", "text_k"])

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=None,
        compute_metrics=None,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=112),
    )

    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir+"/train_save/")
    model.save_pretrained(save_directory=training_args.output_dir)


if __name__ == "__main__":
    train()
