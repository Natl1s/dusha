## Raw data processing

To process data download a raw Dusha dataset (crowd.tar, podcast.tar), untar it to DATASET_PATH, and run the processing script:

    python processing.py -dataset_path  DATASET_PATH 

It processes sound files and creates a folder in DATASET_PATH with precalculated features, aggregates labels, and creates manifest file in jsonl format.


If you want to change the threshold for aggregation run the processing with -threshold flag:

    python processing.py  -dataset_path  DATASET_PATH -threshold THRESHOLD

You can also use tsv format for manifest file:

    python processing.py -dataset_path  DATASET_PATH -tsv  

Force recalculate features:

    python processing.py  -dataset_path  DATASET_PATH  -rf

## Convert JSONL + NPY to LMDB

If your `features/*.npy` are on a slow/external disk and training does too many random reads, convert the dataset into one LMDB file:

    python dataset/lmdb_convert.py \
      --manifest DATASET_PATH/processed_dataset_090/train/train.jsonl \
      --data-root DATASET_PATH/processed_dataset_090 \
      --output DATASET_PATH/processed_dataset_090/train.lmdb

Script supports records with `tensor`, `feature_path`, `hash_id`/`id` (for features), `audio_path`/`wav_path`/`wav` (or `hash_id`/`id` fallback for audio), and stores entries in LMDB as:
- key: `<index>` (bytes)
- value: pickle with:
  - `x`: np.float32 array with acoustic features
  - `y`: int label
  - `id`: sample_id
  - `waveform`: raw mono waveform (`np.float32`) resampled to 16kHz
  - `waveform_sr`: sample rate (`16000`)
  - `text`: transcript (`speaker_text`/`text`/`transcript`/`utterance`)
