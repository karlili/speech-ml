'''
this file downloads common_voice dataset from huggingface and update the content inside the TSV for the sentence transcripts
'''

from datasets import load_dataset, DatasetDict, Dataset, Audio
import pandas as pd



''' definitions'''
# Define the character pairs to update
char_pairs = [
    ('噉', '咁'),
    ('無', '冇'),
    ('呀', '啊'),
    ('瞓', '訓'),
    ('翻', '返'),
    ('噶', '嘅'),

]

# Function to update the characters in a single field
def update_chars(field):
    for original, updated in char_pairs:
        original_field = field

        field = field.replace(original, updated)

        if (original_field != field):
            print(f"|----- {original_field} >>>> {field}")

    return field


''' running logics'''
hub_id = 'poppysmickarlili/common_voice_yue'
audio_dataset_id = "mozilla-foundation/common_voice_17_0"

common_voice = DatasetDict()

common_voice["train"] = load_dataset(audio_dataset_id, "yue", split="train+validation")
common_voice["test"] = load_dataset(audio_dataset_id, "yue", split="test")

# print(common_voice['train'])

train_dataset = common_voice['train']
train_df = train_dataset.to_pandas()

test_dataset = common_voice['test']
test_df = test_dataset.to_pandas()

print(list(train_df), list(test_df))

train_sentence_subset = train_df[['sentence', 'audio', 'path']]
test_sentence_subset = train_df[['sentence', 'audio', 'path']]

# do something to change the sentence
train_sentence_subset['sentence'] = train_sentence_subset['sentence'].apply(update_chars)
test_sentence_subset['sentence'] = test_sentence_subset['sentence'].apply(update_chars)


# Make a new dataset from panda
new_train_dataset = Dataset.from_pandas(train_sentence_subset)
new_train_dataset = new_train_dataset.cast_column("audio", Audio())

new_test_dataset = Dataset.from_pandas(test_sentence_subset)
new_test_dataset = new_test_dataset.cast_column("audio", Audio())

dataset_dict = DatasetDict({'train': new_train_dataset, 'test': new_test_dataset})
print(dataset_dict)

# dataset_dict.push_to_hub(hub_id)