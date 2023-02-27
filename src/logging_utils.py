import json
import os


def delete_files_in_folder(folder):
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))


def write_to_file(filename, input):

    if not os.path.exists("./output_logs"):
        os.makedirs("./output_logs")
    else:
        delete_files_in_folder("./output_logs")

    if isinstance(input, str):
        with open("./output_logs/" + filename, "w") as file:
            file.write(input)
            file.write("\n")
    else:
        with open("./output_logs/" + filename, "w") as file:
            json.dump(input, file, indent=4)
            file.write("\n")
