import json
import os
import shutil


def clear_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))


def write_to_file(filename, input):

    if not os.path.exists("./output_logs/bpmn_structure"):
        os.makedirs("./output_logs/bpmn_structure")

    if isinstance(input, str):
        with open("./output_logs/" + filename, "w") as file:
            file.write(input)
            file.write("\n")
    else:
        with open("./output_logs/" + filename, "w") as file:
            json.dump(input, file, indent=4)
            file.write("\n")
