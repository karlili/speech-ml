"""
 1.
 Before downloading any new dataset,
 make sure to check if it needs to Check and Agrees to the terms first, otherwise the download would fail

 2.
 Before running the training script, make sre you have set the env variable for huggingface_hub
 export HF_HOME="/Volumes/DATA/huggingface/"

 3.
 If the training fails with this exception,
 'RuntimeError: MPS backend out of memory
 (MPS allocated: 23.33 GB, other allocations: 5.32 GB, max allowed: 36.27 GB). Tried to allocate 7.93 GB on private pool.'
 export this variable before running the script, e.g.
 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python training.py


"""

import wandb
import datetime
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
import torch
import evaluate
from dataclasses import dataclass
from transformers import WhisperForConditionalGeneration
from typing import Any, Dict, List, Union

# import ffmpeg
# import librosa

now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")

dataset_name = "mozilla-foundation/common_voice_17_0"
language_to_train = 'yue'

# model_name = 'openai/whisper-small'
model_name = 'poppysmickarlili/whisper-small-cantonese'
audio_dataset_id = 'poppysmickarlili/common_voice_yue'
# setting up the configurations in wandb '

config = {
    # "model_name": "whisper-small-cantonese_" + now,
    "model_name": "whisper-small-cantonese",
    "gradient_accumulation_steps": 2,  # increase by 2x for every 2x decrease in batch size
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "max_steps": 4000,

    #gradient checkpointing and use_reentrant are related to each other
    #if gradient checkpoint is True, set use_reentrant to True to potentially reducing memory usage
    "gradient_checkpointing": True,
    "use_reentrant": True,
    "use_cache": False,

    # evaluation strategy can be 'no', 'steps' or 'epoch'
    "evaluation_strategy": "steps",

    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,

    "predict_with_generate": False,
    "generation_max_length": 225,

    "save_steps": 1000,
    "eval_steps": 1000,
    "logging_steps": 25,
    "metric_for_best_model": "wer",
    "num_train_epochs": 2,
}

wandb.init(project="language-x-change", config=config)

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    model_name,
)  # start with the whisper small checkout
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="chinese", task="transcribe", )
processor = WhisperProcessor.from_pretrained(model_name, language="chinese", task="transcribe", )

metric = evaluate.load("wer")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = \
        processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    processor.tokenizer.set_prefix_tokens(language="cantonese", task="transcribe")
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    wandb.log({"wer": wer})
    return {"wer": wer}


def main():

    # Preparing Data -- Whisper expecting the audio to be at sampling rate @16000 - this is just to make sure the
    # sampling rate fits whisper's training Since our input audio is sampled at 48kHz, we need to downsample it to
    # 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the
    # Whisper model.



    # Load Dataset from common-voice

    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(audio_dataset_id, split="train", trust_remote_code=True)
    common_voice["test"] = load_dataset(audio_dataset_id, split="test", trust_remote_code=True)


    # common_voice = common_voice.remove_columns(
    #     ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    '''-------------------------------------------------'''

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

    print("Preparing dataset")

    print("Training -------")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name
    ).to('cuda')

    # model.config.suppress_tokens = []
    model.generation_config.language = 'chinese'
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None


    # data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    training_args = Seq2SeqTrainingArguments(
        output_dir="./drive/MyDrive/whisper-tuning/model----working/" + wandb.config['model_name'],
        # change to a repo name of your choice

        per_device_eval_batch_size=wandb.config["per_device_eval_batch_size"],
        per_device_train_batch_size=wandb.config["per_device_train_batch_size"],
        gradient_accumulation_steps=wandb.config["gradient_accumulation_steps"],
        learning_rate=wandb.config["learning_rate"],

        warmup_steps=wandb.config["warmup_steps"],
        max_steps=wandb.config["max_steps"],
        gradient_checkpointing=wandb.config["gradient_checkpointing"],
        evaluation_strategy=wandb.config["evaluation_strategy"],
        save_steps=wandb.config["save_steps"],
        eval_steps=wandb.config["eval_steps"],
        logging_steps=wandb.config["logging_steps"],
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard", "wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    trainer.train()
    wandb.finish()

    '''pushing the model to huggingface hub'''

    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_16_0",
        "dataset": "Common Voice 16.0",  # a 'pretty' name for the training dataset
        "dataset_args": "config: yue, split: test",
        "language": "yue",
        "model_name": "Whisper Small Cantanese",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
    }

    trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
