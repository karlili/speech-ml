'''
Install the following dependencies before running the script

- pip install transformers accelerate
- pip install json datetime
- pip install torch torchvision torchaudio

'''

import torch
import json
import datetime

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id='openai/whisper-large-v3'

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=20,
    batch_size=16,
    return_timestamps=True,
    # torch_dtype=torch_dtype,
    # device= torch.device('mps'),
    generate_kwargs={"language": "cantonese"}
)

pipe.tokenizer.get_decoder_prompt_ids(task="transcribe")

# this is the place you modify your input - the name of the mp3 file you want to run

import os
import pandas as pd
# directory_path = './voice'

directory_path = '<source_folder_for_test_data>>'

run_timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%s_%f")
run_folder = f"./output/run_{run_timestamp}"
run_result_csv = run_folder + "/result.csv"


os.mkdir(run_folder)
file_list = os.listdir(directory_path)
result_df = pd.DataFrame(columns=['using_model', 'file_name','transcript','json_original'])
total_num_of_files = len(file_list)
for index, file in enumerate(file_list):
    # print(file)
    if not file.endswith('.mp3'):
        continue

    result = pipe(directory_path+'/'+file)

    # then it will write the response in a json file named as the current date time

    json_object = json.dumps(result, ensure_ascii=False)
    with open(run_folder+'/'+file+".json", "w", encoding='utf8') as f:
        f.write(json_object)

    # also it will print out the result in the following output block
    row_result = pd.DataFrame([{"using_model": model_id, "file_name": file, "transcript": result['text'], "json_original": json_object}])
    print(f"{index+1} of {total_num_of_files} | {file} -- {result['text']}")
    result_df = pd.concat([result_df, row_result], ignore_index=True)

print(result_df)
# Output the results into a csv file
result_df.to_csv(run_result_csv, index=False)