import pandas as pd
from difflib import SequenceMatcher, Differ


directory_path = '<directory_path>'
source_csv_path = f"{directory_path}/Uploaded-Sentences-Audios.csv"
target_csv_path = f"{directory_path}/sentence_transcript_result.csv"
source_csv_df = pd.read_csv(source_csv_path)
target_csv_df = pd.read_csv(target_csv_path)

'''make sure the two csv have the same file_name in header'''
source_csv_df.rename(columns={
    'sentences_filenames': 'file_name',
    'canto_sentences': 'original_text'
}, inplace=True)

target_csv_df.rename(columns={
    'file_name': 'file_name'
}, inplace=True)


def calculate_difference(ds):
    # print(f">>> {ds['original_text']}")
    s = SequenceMatcher(None, ds['transcript'], ds['original_text'])
    d = Differ()
    match_ratio = s.ratio()
    difference = d.compare(ds['original_text'], ds['transcript'])

    new_column = {'match_ratio': match_ratio}
    return match_ratio


# def analysis_part_of_speech(ds):
#     # print(ds['original_text'])
#     # text = pycantonese.parse_text()
#     pos = pycantonese.characters_to_jyutping(ds['original_text'])
#     print(f"{pos}")
#     return pos


'''combine the two dataframe into one and merge with the common key'''
combined_df = pd.merge(source_csv_df, target_csv_df, on='file_name')
combined_df['match_ratio'] = combined_df.apply(lambda x: calculate_difference(x), axis=1)
# combined_df['jyutping'] = combined_df.apply(lambda x: analysis_part_of_speech(x), axis=1)

print(combined_df.head())
combined_df.to_csv(f"./combined_result.csv", index=False)
