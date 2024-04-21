import os

DATA_PATH = "data/SentiCap"

def processed_tensors_management(list_dirs: list[str]) -> None:

    # Paths for the source and destination of the files
    master_directory = f"{DATA_PATH}/Processed_tensors"

    if not os.path.exists(master_directory):
        os.mkdir(master_directory)

    for directory in list_dirs:

        d = f"{master_directory}/{directory}"
        if not os.path.exists(d):
            os.mkdir(d)

