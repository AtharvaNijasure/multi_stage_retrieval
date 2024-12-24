import os
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, PreTrainedModel, GPTNeoXForSequenceClassification, AutoTokenizer
from transformers.file_utils import ModelOutput
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft import PeftConfig, get_peft_model
from peft import get_peft_model_state_dict

import logging
import random
from transformers.trainer import Trainer

from huggingface_hub.hf_api import HfFolder 

from rerank_arguments import ModelArguments , DataArguments

from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field

HfFolder.save_token('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')
logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class RerankerModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForSequenceClassification

    def __init__(self, hf_model: PreTrainedModel, train_batch_size: int = None):
        super().__init__()
        self.config = hf_model.config
        self.hf_model = hf_model
        self.train_batch_size = train_batch_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_batch_size:
            self.register_buffer(
                'target_label',
                torch.zeros(self.train_batch_size, dtype=torch.long, device=self.hf_model.device)
            )
        for name, param in self.hf_model.named_parameters():
            # for some reason, ds zero 3 left some weights empty
            if 'modules_to_save' in name and param.numel() == 0:
                logger.warning(f'parameter {name}, shape {param.shape} is empty')
                param.data = nn.Linear(self.hf_model.config.hidden_size, 1).weight.data
                logger.warning('{} data: {}'.format(name, param.data.cpu().numpy()))

    def forward(self, pair: Dict[str, Tensor] = None):
        ranker_logits = self.hf_model(**pair, return_dict=True).logits
        if self.train_batch_size:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            loss = self.cross_entropy(grouped_logits, self.target_label)
            return RerankerOutput(
                loss = loss,
                scores = ranker_logits
            )

        return RerankerOutput(
            loss = None,
            scores = ranker_logits
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        # if llama model uncomment next line
        # self.hf_model.base_model.model.gradient_checkpointing_enable(**kwargs)
        # if pythia model uncomment next line
        self.hf_model.base_model.gradient_checkpointing_enable(**kwargs)
        # self.hf_model.gradient_checkpointing_enable(**kwargs)


    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name_or_path,
            **hf_kwargs,
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.SEQ_CLS,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                hf_model=lora_model,
                train_batch_size=train_args.per_device_train_batch_size,
            )
        else:
            model = cls(
                hf_model=base_model,
                train_batch_size=train_args.per_device_train_batch_size,
            )
        return model

    @classmethod
    def load(cls,
            model_name_or_path: str,
            lora_name_or_path: str = None,
            **hf_kwargs):
        
        # Load LoRA config first
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
    
        base_model = cls.TRANSFORMER_CLS.from_pretrained(peft_config.base_model_name_or_path, num_labels=1, **hf_kwargs, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                hf_model=lora_model,
            )
        else:
            model = cls(
                hf_model=base_model,
            )
        return model

    def save(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
        
        

class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RerankerTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

        # if is_deepspeed_zero3_enabled():
        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'hf_model.'
        assert all(
            k.startswith(prefix) or k == "target_label"
            for k in state_dict.keys()
        ), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        lora_state_dict = get_peft_model_state_dict(self.model.hf_model, state_dict)
        if self.args.process_index <= 0:
            torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
            print(f"Save adapter model at {output_dir}")


    def compute_loss(self, model, inputs, num_items_in_batch = None):
        outputs = model(inputs)
        loss = outputs.loss / num_items_in_batch if num_items_in_batch else outputs.loss
    
        return loss


@dataclass
class RerankerTrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[List[str]]):
        """
        Collate function for training.
        :param features: list of pairs
        :return: tokenized pairs
        """
        all_pairs = []
        for pairs in features:
            all_pairs.extend(pairs)
        
        tokenized_pairs = self.tokenizer(
            all_pairs,
            padding=False, 
            truncation=True,
            max_length=self.data_args.rerank_max_len-1 if self.data_args.append_eos_token else self.data_args.rerank_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Option 1: Use eos_token as pad_token

        if self.data_args.append_eos_token:
            tokenized_pairs['input_ids'] = [p + [self.tokenizer.eos_token_id] for p in tokenized_pairs['input_ids']]
        
        pairs_collated = self.tokenizer.pad(
            tokenized_pairs,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return pairs_collated


@dataclass
class RerankerInferenceCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (query_id, text_id, pair) tuples
        """
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        pairs = [x[2] for x in features]
        collated_pairs = self.tokenizer(
            pairs,
            padding=False, 
            truncation=True,
            max_length=self.data_args.rerank_max_len-1 if self.data_args.append_eos_token else self.data_args.rerank_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_pairs['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_pairs['input_ids']]
        collated_pairs = self.tokenizer.pad(
            collated_pairs,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return query_ids, text_ids, collated_pairs
    



def format_pair(query: str, passage: str, title: str, query_prefix: str, passage_prefix: str):
    title = title.replace('-', ' ').strip()
    return f'{query_prefix} {query} {passage_prefix} {title} {passage}'.strip()



class RerankerTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None, subset_ratio = None):
        self.data_args = data_args
        self.dataset = load_dataset( # train_data # dataset
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        # # Sample subset if ratio provided
        if subset_ratio:
            total_size = len(self.dataset)
            subset_size = int(total_size * subset_ratio)
            indices = random.sample(range(total_size), subset_size)
            self.train_data = self.dataset.select(indices)
            logger.info(f"Sampled {subset_size} examples ({subset_ratio*100}% of {total_size})")
        else :
            self.train_data = self.dataset
        
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_pair = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_pair.append(format_pair(query, pos_psg['text'], pos_psg['title'], self.data_args.query_prefix, self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_pair.append(format_pair(query, neg_psg['text'], neg_psg['title'], self.data_args.query_prefix, self.data_args.passage_prefix))

        return formated_pair



class RerankerInferenceDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.inference_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            trust_remote_code=True
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.inference_data = self.inference_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.inference_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        example = self.inference_data[item]
        query_id = example['query_id']
        query = example['query']
        text_id = example['docid']
        text = example['text']
        title = example['title']
        return query_id, text_id, format_pair(query, text, title, self.data_args.query_prefix, self.data_args.passage_prefix) 

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default='./pythiatest', metadata={"help": "huggingface dataset name"}
    )

def main():

    
    tokenizer_name = "EleutherAI/pythia-6.9b"
    model_name = "EleutherAI/pythia-6.9b"
    cache_dir = None
    output_dir = "aug_model_pythia_6_9"
    save_steps = 50 
    dataset_name = "Tevatron/msmarco-passage-aug"
    query_prefix = "query: " 
    passage_prefix = "document: " 
    bf16 = True
    per_device_train_batch_size  = 16 
    gradient_checkpointing  = True
    train_group_size = 16 
    learning_rate = 1e-4 
    rerank_max_len = ( 32 + 164 )
    num_train_epochs = 1 
    logging_steps = 100 
    gradient_accumulation_steps = 4


    tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    # if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'right'


    
        

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    # manual_args = {
    #     "tokenizer_name": "EleutherAI/pythia-2.8b",
    #     "model_name_or_path": "EleutherAI/pythia-2.8b",
    #     "cache_dir": None,
    #     "dataset_name": "Tevatron/msmarco-passage-aug",
    #     "query_prefix": "query: ",
    #     "passage_prefix": "document: ",
    #     "output_dir": "aug_model_pythia_2_8",
    #     "save_steps": 50,
    #     "bf16": True,
    #     "per_device_train_batch_size": 1,
    #     "gradient_checkpointing": True,
    #     "train_group_size": 1,
    #     "learning_rate": 1e-4,
    #     "rerank_max_len": 32 + 164,
    #     "num_train_epochs": 1,
    #     "logging_steps": 100,
    #     "gradient_accumulation_steps": 4,
    # }

    # model_args, data_args, training_args = parser.parse_dict(manual_args)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = RerankerModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )

    set_seed(training_args.seed)
    train_dataset = RerankerTrainDataset(data_args)
    train_collator = RerankerTrainCollator(data_args, tokenizer)


    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator
    )
    train_dataset.trainer = trainer


    print(f"Done till here - model: {trainer.model}")

    torch.cuda.empty_cache()
    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)




if __name__ == "__main__":
    main()



'''


deepspeed --master_port 60000 --num_gpus 8 --module reranking_finetune.train_pythia \
  --deepspeed /project/pi_allan_umass_edu/anijasure/rankllama3/reranking_finetune/deepspeed/ds_zero3_config.json \
  --output_dir aug_model_pythia_6_9 \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --model_name_or_path EleutherAI/pythia-6.9b \
  --save_steps 50 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --train_group_size 1 \
  --learning_rate 1e-4 \
  --rerank_max_len $(( 32 + 164 )) \
  --num_train_epochs 1 \
  --logging_steps 100 \
  --gradient_accumulation_steps 1


'''