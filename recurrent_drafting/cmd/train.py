#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

"""
Training arguments:
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
"""

import math
import multiprocessing
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader

from recurrent_drafting.configuration_drafter import DrafterConfig
from recurrent_drafting.modeling_drafter import Drafter
from recurrent_drafting.train import data
from recurrent_drafting.train.loss import drafter_loss
from recurrent_drafting.train.model import ReDrafter


@dataclass
class ModelArguments:
    llm_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    drafter_name_or_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = None
    model_max_length: int = 2048
    drafter_predict_n_tokens: int = 5
    drafter_top_k: int = 5
    drafter_num_layers: int = 1
    include_inputs_for_metrics: bool = True
    phase: str = "train"
    rnn: bool = False
    output_dir: str = "./output"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_tokenizer(model_args, training_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def generate_drafter_config_from_base(llm, training_args):
    return DrafterConfig(
        vocab_size=llm.lm_head.weight.shape[0],
        hidden_size=llm.lm_head.weight.shape[-1],
        exit_dim=2 * llm.lm_head.weight.shape[-1],
        num_draft_layers=training_args.drafter_num_layers,
        rnn=training_args.rnn,
    )


def train(model_args, training_args):
    tokenizer = get_tokenizer(model_args, training_args)
    train_dataset = datasets.load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train").map(
        lambda x: data.sharegpt_record_to_vicuna_training_instance(x, tokenizer),
        num_proc=multiprocessing.cpu_count(),
    )

    config = transformers.AutoConfig.from_pretrained(model_args.llm_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.llm_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    for param in llm.base_model.parameters():
        param.requires_grad = False

    drafter_config = generate_drafter_config_from_base(llm, training_args)
    drafter = Drafter(drafter_config)
    redrafter = ReDrafter(llm, drafter).to(training_args.device)

    # Prepare DataLoader
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        redrafter.drafter.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    num_training_steps = len(train_loader) * training_args.num_train_epochs // training_args.gradient_accumulation_steps
    global_step = 0
    redrafter.train()
    for epoch in range(training_args.num_train_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(training_args.device)
            logits = redrafter(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                next_n=training_args.drafter_predict_n_tokens,
            )
            loss, log, eval_log = drafter_loss(
                logits, batch["labels"], training_args.drafter_predict_n_tokens, training_args.drafter_top_k
            )
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % training_args.logging_steps == 0:
                    print(f"Epoch {epoch+1} Step {global_step}: Loss {epoch_loss/(step+1):.4f}")
                if global_step % training_args.save_steps == 0:
                    drafter.save_pretrained(training_args.output_dir)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss/(step+1):.4f}")

    drafter.save_pretrained(training_args.output_dir)
    print(f"Training complete. Drafter saved to {training_args.output_dir}")


def eval(model_args, training_args):
    print("Evaluation is not implemented in this explicit loop version.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name_or_path", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--drafter_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--drafter_predict_n_tokens", type=int, default=5)
    parser.add_argument("--drafter_top_k", type=int, default=5)
    parser.add_argument("--drafter_num_layers", type=int, default=1)
    parser.add_argument("--rnn", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    model_args = ModelArguments(
        llm_name_or_path=args.llm_name_or_path,
        drafter_name_or_path=args.drafter_name_or_path,
    )
    training_args = TrainingArguments(
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        drafter_predict_n_tokens=args.drafter_predict_n_tokens,
        drafter_top_k=args.drafter_top_k,
        drafter_num_layers=args.drafter_num_layers,
        rnn=args.rnn,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    train(model_args, training_args)
