import argparse
import configparser

from process_bpmn_data import generate_graph_pdf, process_text


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--text", help="Textual description of a process")
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a file containing a textual description of a process",
    )
    parser.add_argument(
        "-n",
        "--notebook",
        help="If true, will output a .gv file that can be rendered in a notebook",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="The OpenAI model to use",
        choices=["gpt-3.5-turbo", "gpt-4"],
        default="gpt-3.5-turbo",
    )

    args = parser.parse_args()

    if not (args.text or args.file):
        parser.error("Please provide a text or a file")

    config = configparser.ConfigParser()
    config.read("src\config.ini")
    config["OPENAI"] = {}
    config["OPENAI"]["model"] = args.model
    with open("src\config.ini", "w") as configfile:
        config.write(configfile)

    print(f"Using OpenAI model: {config['OPENAI']['model']}")

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r") as f:
            text = f.read()

    return (text, args.notebook)


if __name__ == "__main__":

    text, notebook = parse_arguments()

    output = process_text(text)

    generate_graph_pdf(output, notebook)
