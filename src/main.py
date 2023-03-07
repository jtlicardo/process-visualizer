import argparse

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

    args = parser.parse_args()

    if not (args.text or args.file):
        parser.error("Please provide a text or a file")

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r") as f:
            text = f.read()

    return (text, args.notebook)


if __name__ == "__main__":

    text, notebook = parse_arguments()
    
    try:
        output = process_text(text)
    except:
        print("Error when processing text")
        exit()
    try:
        generate_graph_pdf(output, notebook)
    except:
        print("Error when generating graph")
