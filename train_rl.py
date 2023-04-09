# coding:utf-8

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="bigscience/bloomz-1b1", metadata={"help": "the model name"})
    logging_dir: Optional[str] = field(default="data/log/", metadata={"help": "log with accelerator_kwargs"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    dataset_name: Optional[str] = field(default="data/reward_train.json", metadata={"help": "the dataset file"})
    seed: Optional[int] = field(default=307, metadata={"help": "the seed"})
    reward_model_name: Optional[str] = field(default="model/reward_model", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="model/rl_model", metadata={"help": "the output"})


def build_dataset(model_name, dataset_name, input_min_text_length=2, input_max_text_length=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_json(path_or_paths=dataset_name)
    ds = ds.rename_columns({"prompt": "review"})
    ds = ds.filter(lambda x: len(x["review"]) >= 2, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        accelerator_kwargs={"logging_dir": script_args.logging_dir}
    )
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    dataset = build_dataset(script_args.model_name, script_args.dataset_name)
    set_seed(script_args.seed)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    # reward model
    rw_model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    output_min_length = 1
    output_max_length = 256
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + " " + tokenizer.bos_token + " " + r for q, r in zip(batch["query"], batch["response"])]
        outputs = rw_model(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]) for output in outputs]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    model.save_pretrained(script_args.output_dir)


if __name__ == "__main__":
    train()
