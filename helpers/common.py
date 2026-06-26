from pathlib import Path

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_weights_dir']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])