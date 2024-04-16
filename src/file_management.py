import os
import shutil
import random


def emotion_management_1():
    '''Initially, the emotions are stored both in txt and anvil files. This
    function filters only the text files. Additionally, given that the
    utterances are rated by multiple annotators,
    we only store the first rating.
    '''
    emot_dir = "data/Not_Sorted_Emotion/Emotion"
    destination_dir = "data/Emotion"

    # Create destination directory
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

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


def emotion_management_2():
    '''After emotion_management_1, the categorization of the emotions 
    is stored in text files, corresponding one to each dialog, out of the
    28 available. This function reads the text files and stores the
    utterances in separate files, each containing the emotion of the
    utterance. Additionally, the emotions are assigned a number, starting
    from 0. The corresponding mapping is returned.

    Returns:
        dict: A dictionary mapping emotions (strings) to numbers.
    '''
    i = 0
    emotion_dict = {}

    list_dir = os.listdir("data/Emotion")

    if not os.path.exists("data/Emotion/Utterances"):
        os.makedirs("data/Emotion/Utterances")

    for file in list_dir:
        with open(os.path.join("data/Emotion", file), "r") as f:
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
                    emotion = i
                    i += 1
                else:
                    emotion = emotion_dict[emotion]

                # Store the utterance and emotion
                out_file = os.path.join("data/Emotion/Utterances", utterance + ".txt")
                with open(out_file, "w") as out:
                    out.write(str(emotion))

    return emotion_dict


def audio_management():
    '''The audio is already segmented in utterances, but each dialog
    has its own folder. This function moves all the audio files to a
    single directory
    '''
    audio_dir = "data/Not_Sorted_Audio"
    destination_dir = "data/Audio"

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


def text_management():
    '''With text, a similar problem arises as with emotions. Initially,
    the text is stored in a single file for each dialog. This function
    reads the text files and stores the utterances in separate files,
    each containing the transcript of the utterance. Therefore, each 
    file (with the id of the utterance) contains the transcript of the
    utterance.
    '''
    text_dir = "data/Not_Sorted_Text"
    destination_dir = "data/Text"

    list_texts = os.listdir(text_dir)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for dialog in list_texts:
        with open(os.path.join(text_dir, dialog), "r") as f:
            for line in f.readlines():
                utterance, transcript = line.split(":", 2)

                # Utterance is of the format "Ses01F_impro01_F000 [1.2900-3.2100]", so we need to extract the first part
                utterance = utterance.split(" ")[0]

                # Store the utterance and transcript
                out_file = os.path.join(destination_dir, utterance + ".txt")

                # print(out_file)
                if out_file == "data/Text\M.txt" or out_file == "data/Text\F.txt":
                    continue

                with open(out_file, "w") as out:
                    out.write(transcript.strip())


def divide_data(train_prop: float = 0.8):
    '''Divides the data into training and testing sets. The training
    proportion is given as an argument.

    Args:
        train_prop (float, optional): The proportion of the data that
            will be used for training. Defaults to 0.8.
    '''
    audio_dir = "data/Audio"
    text_dir = "data/Text"
    emotion_dir = "data/Emotion/Utterances"

    audio_files = os.listdir(audio_dir)
    # text_files = os.listdir(text_dir)
    # emotion_files = os.listdir(emotion_dir)

    # Create the directories
    train_dir = "data/Training"
    test_dir = "data/Testing"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for mode in ["Training", "Testing"]:
        for modality in ["Audio", "Text", "Emotion"]:
            os.makedirs(f"data/{mode}/{modality}", exist_ok=True)	

    # Choose the training utterances
    file_mapping = {}
    for file in audio_files:
        utterance = file.split(".")[0]
        file_mapping[utterance] = {
            "Audio": file,
            "Text": utterance + ".txt",
            "Emotion": utterance + ".txt"
        }

    # Shuffle the utterances
    utterance_ids = list(file_mapping.keys())
    print(utterance_ids[:10])
    random.shuffle(utterance_ids)

    # Number of training utterances
    num_train = int(train_prop * len(utterance_ids))

    # Move the utterances to the corresponding directories
    for i, utterance in enumerate(utterance_ids):
        source_dir = train_dir if i < num_train else test_dir
        for modality in ["Audio", "Text", "Emotion"]:
            print("file_mapping: ", file_mapping[utterance][modality])
            source_path = os.path.join(source_dir, modality, file_mapping[utterance][modality])
            print("Source dir: ", source_dir)
            print("Modality: ", modality)
            print("Utterance: ", utterance)
            input("Press Enter to continue...")
            destination_path = os.path.join(source_dir, modality)
            shutil.move(source_path, destination_path)


if __name__ == "__main__":
    # Delete the old files
    # if os.path.exists("data/Emotion/Utterances"):
    #     shutil.rmtree("data/Emotion/Utterances")
    # emotion_management_1()
    # emotions = emotion_management_2()
    # print(emotions)
    # audio_management()

    # if os.path.exists("data/Text"):
    #     shutil.rmtree("data/Text")
    # text_management()

    divide_data()
