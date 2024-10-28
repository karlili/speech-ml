from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

# root_path = '/Users/kenny/Google Drive/My Drive/Team/Training reference/'
root_path = '/Users/kenny/Downloads/upload'

def make_chunks_from_slience(root_path, file):
    full_file_path = f'{root_path}/{file}'
    sound_file = AudioSegment.from_mp3(full_file_path)
    audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-40)

    for i, chunk in enumerate(audio_chunks):
        out_file = f"{root_path}/chunks/{file}_chunk_{i}.mp3"
        print("exporting", out_file)
        chunk.export(out_file, format="wav")


def make_equal_chunks(chunk_length_ms, root_path, file):
    # chunk_length_ms = 10000  # 10 seconds

    full_file_path = f'{root_path}/{file}'
    sound_file = AudioSegment.from_mp3(full_file_path)

    # Slice audio and export chunks
    for i in range(0, len(sound_file), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        out_file = f"{root_path}/chunks/{file}_chunk{i // chunk_length_ms}.mp3"
        chunk.export(out_file, format="mp3")


# os.mkdir(root_path'/chunks')
file_list = os.listdir(root_path)
total_num_of_files = len(file_list)
for index, file in enumerate(file_list):
    # print(file)
    if not file.endswith('.mp3'):
        continue

    make_chunks(root_path, file)


