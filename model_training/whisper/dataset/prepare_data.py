'''
Install the following dependencies before running the scripts

!pip install datasets pandas
'''


'''
Configuration Data setup
'''
hub_id = '<hub_id>'
audio_path_location = '<audio_path_location>'
csv_path_location = '<audio_path_location>/canto-vocab-test.csv'


'''
Main Logics
'''
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import Audio, ClassLabel

# Load the CSV file
df = pd.read_csv(csv_path_location)

# Create a dataset from the CSV file
dataset = Dataset.from_pandas(df)


# Define a function to map the text to audio
def map_text_to_audio(ds):
    print(ds)

    audio_paths = f"{audio_path_location}/{ds['audio_file']}"
    return {'audio': audio_paths, 'canton_vocab': ds['canton_vocab']}


# Apply the mapping function to the dataset
dataset = dataset.map(map_text_to_audio)

# Get the actual audio file from the given path
dataset = dataset.cast_column("audio", Audio())

# Create a dataset dictionary
dataset_dict = DatasetDict({'train': dataset})
# print(dataset_dict)

# Upload the dataset to huggingface
dataset_dict.push_to_hub(hub_id)