import os
import shutil

DATA_PATH = "data/IEMOCAP"
PROCESSED_TENSORS_PATH = "Processed_tensors"


def emotion_management_1() -> None:
    """
    Initially, the emotions are stored both in txt and anvil files. This
    function filters only the text files. Additionally, given that the
    utterances are rated by multiple annotators,
    we only store the first rating.
    """

    # The directory containing the emotions' files
    emot_dir = f"{DATA_PATH}_PATH/Not_Sorted_Emotion/Emotion"
    # The directory where the individual .txt files will be stored
    destination_dir = f"{DATA_PATH}/Emotion"

    # Create destination directory
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Dictionary to take only one instance of each dialog
    scenes = {}
    for file in os.listdir(emot_dir):
        # Check if text file
        if file.endswith(".txt"):
            name_scene = "_".join(file.split("_")[:-2])

            # Check if scene is already stored
            if name_scene not in scenes:
                # Move file to destination directory
                source_path = os.path.join(emot_dir, file)
                destination_path = os.path.join(destination_dir, name_scene + ".txt")
                shutil.move(source_path, destination_path)

                scenes[name_scene] = True


def emotion_management_2() -> dict[str, int]:
    """
    After emotion_management_1, the categorization of the emotions
    is stored in text files, corresponding one to each dialog, out of the
    28 available. This function reads the text files and stores the
    utterances in separate files, each containing the emotion of the
    utterance. Additionally, the emotions are assigned a number, starting
    from 0. The corresponding mapping is returned.

    Returns:
        A dictionary mapping emotions (strings) to numbers.
    """

    i = 0
    emotion_dict = {}

    list_dir = os.listdir(f"{DATA_PATH}/Emotion")

    if not os.path.exists(f"{DATA_PATH}/Emotion/Utterances"):
        os.makedirs(f"{DATA_PATH}/Emotion/Utterances")

    for file in list_dir:
        with open(os.path.join(f"{DATA_PATH}/Emotion", file), "r") as f:
            # The file is full of utterances
            for line in f.readlines():
                # Extract the parts of the utterance
                parts = line.split(":", 2)

                # Eliminate the trailing whitespace
                utterance = parts[0].strip()

                # Extract the emotion
                emotion = parts[1].split(";")[0].strip()

                # Store the emotion
                if emotion not in emotion_dict:
                    emotion_dict[emotion] = i
                    emotion = str(i)
                    i += 1
                else:
                    emotion = str(emotion_dict[emotion])

                # Store the utterance and emotion
                out_file = os.path.join(
                    f"{DATA_PATH}/Emotion/Utterances", utterance + ".txt"
                )
                with open(out_file, "w") as out:
                    out.write(str(emotion))

    return emotion_dict


def audio_management() -> None:
    """
    The audio is already segmented in utterances, but each dialog
    has its own folder. This function moves all the audio files to a
    single directory
    """

    audio_dir = f"{DATA_PATH}/Not_Sorted_Audio"
    destination_dir = f"{DATA_PATH}/Audio"

    list_audios = os.listdir(audio_dir)

    # Create destination directory
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for dialog in list_audios:
        # Check if the file is a dialog
        folder_path = os.path.join(audio_dir, dialog)
        if os.path.isdir(folder_path):
            # Iterate over the files in the dialog
            for file in os.listdir(folder_path):
                # Check if the file is an audio file
                if file.endswith(".wav"):
                    # Move the file to the destination directory
                    source_path = os.path.join(folder_path, file)
                    destination_path = os.path.join(destination_dir, file)
                    shutil.move(source_path, destination_path)


def text_management() -> None:
    """
    With text, a similar problem arises as with emotions. Initially,
    the text is stored in a single file for each dialog. This function
    reads the text files and stores the utterances in separate files,
    each containing the transcript of the utterance. Therefore, each
    file (with the id of the utterance) contains the transcript of the
    utterance.
    """

    # Paths for the source and destination of the files
    text_dir = f"{DATA_PATH}/Not_Sorted_Text"
    destination_dir = f"{DATA_PATH}/Text"

    # Take all text documents
    list_texts = os.listdir(text_dir)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate over the different dialogs
    for dialog in list_texts:
        with open(os.path.join(text_dir, dialog), "r") as f:
            # Extract individual utterances
            for line in f.readlines():
                utterance, transcript = line.split(":", 2)

                # Utterance is of the format "Ses01F_impro01_F000
                # [1.2900-3.2100]", so we need to extract only the title
                utterance = utterance.split(" ")[0]

                # Store the utterance and transcript
                out_file = os.path.join(destination_dir, utterance + ".txt")

                # Some problems arose given that some utterance isn't correctly
                # formatted (in the format <utterance> <time> <dialog>)
                if (
                    out_file == f"{DATA_PATH}/Text\M.txt"
                    or out_file == f"{DATA_PATH}/Text\F.txt"
                ):
                    continue

                # Write the phrase
                with open(out_file, "w") as out:
                    out.write(transcript.strip())


def processed_tensors_management(list_dirs: list[str]) -> None:
    """
    This function checkes wheter or not the paths for the storing of tensors
    realated to the IEMCAP have been created. If not, it cretes them.

    Args:
        list_dirs: the name of the sub directories in the tensors folders. One\
            for every group of data tensors
    """

    # Paths for the source and destination of the files
    master_directory = DATA_PATH + '/' + PROCESSED_TENSORS_PATH

    if not os.path.exists(master_directory):
        os.mkdir(master_directory)

    for directory in list_dirs:
        d = f"{master_directory}/{directory}"
        if not os.path.exists(d):
            os.mkdir(d)


if __name__ == "__main__":
    # Delete the old directories
    if os.path.exists(f"{DATA_PATH}/Emotion/Utterances"):
        shutil.rmtree(f"{DATA_PATH}/Emotion/Utterances")
    if os.path.exists(f"{DATA_PATH}/Text"):
        shutil.rmtree(f"{DATA_PATH}/Text")
    if os.path.exists(f"{DATA_PATH}/Audio"):
        shutil.rmtree(f"{DATA_PATH}/Audio")

    # Filter emotion texts and store the categories' dictionary
    emotion_management_1()
    emotions = emotion_management_2()
    with open(f"{DATA_PATH}/emotions.txt", "w") as f:
        for emotion, value in emotions.items():
            f.write(f"{emotion}: {value}\n")

    # Move the audio files
    audio_management()

    # Individualise utterances' transcriptions
    text_management()
