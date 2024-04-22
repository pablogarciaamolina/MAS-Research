import os

DATA_PATH = "data/SentiCap"
PROCESSED_TENSORS_PATH = "Processed_tensors"


def processed_tensors_management(list_dirs: list[str]) -> None:
    """
    This function checkes wheter or not the paths for the storing of tensors
    realated to the IEMCAP have been created. If not, it cretes them.

    Args:
        list_dirs: the name of the sub directories in the tensors folders. One\
            for every group of data tensors
    """

    # Paths for the source and destination of the files
    master_directory = DATA_PATH + PROCESSED_TENSORS_PATH

    if not os.path.exists(master_directory):
        os.mkdir(master_directory)

    for directory in list_dirs:
        d = f"{master_directory}/{directory}"
        if not os.path.exists(d):
            os.mkdir(d)
