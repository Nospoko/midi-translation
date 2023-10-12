from huggingface_hub import hf_hub_download

FILENAME_VELOCITY = "midi-transformer-2023-10-03-08-34.pt"
FILENAME_DSTART = "midi-transformer-2023-10-06-11-55.pt"
FILENAME_SEQ_DSTART = "128v-midi-transformer-2023-10-11-09-38.pt"

hf_hub_download(
    repo_id="wmatejuk/midi-velocity-transformer",
    filename=FILENAME_VELOCITY,
    local_dir="checkpoints/velocity",
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id="wmatejuk/midi-dstart-transformer",
    filename=FILENAME_DSTART,
    local_dir="checkpoints/dstart",
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id="wmatejuk/midi-dstart-transformer",
    filename=FILENAME_SEQ_DSTART,
    local_dir="checkpoints/dstart",
    local_dir_use_symlinks=False,
)
