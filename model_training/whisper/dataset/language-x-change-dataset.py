from datasets import load_dataset, DatasetDict, Dataset, Audio
import pandas as pd


audio_dataset_id='<audio_dataset_id>'

sentence_dataset = load_dataset(audio_dataset_id,  split="train")

print(sentence_dataset)

sentence_df = sentence_dataset.to_pandas()
print(sentence_df)
