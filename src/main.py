import argparse

from coreference_resolution.coref import resolve_references
from graph_generator import GraphGenerator
from logging_utils import delete_files_in_folder, write_to_file
from process_bpmn_data import *


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


def get_model_outputs(text):

    print("Getting model outputs...\n")
    data = query({"inputs": text, "options": {"wait_for_model": True}})
    write_to_file("model_output.txt", data)

    if "error" in data:
        print("Error when getting model outputs:", data["error"])
        return None

    return data


def extract_all_entities(data):

    print("Extracting entities...\n")
    agents = extract_entities("AGENT", data)
    tasks = extract_entities("TASK", data)
    conditions = extract_entities("CONDITION", data)
    return (agents, tasks, conditions)


def get_sentence_data(text):

    sents = get_sentences(text)
    sents_data = create_sentence_data(sents)
    write_to_file("sentences.txt", sents_data)
    return sents_data


def process_text(text):

    print("\nInput text:\n" + text)

    if should_resolve_coreferences(text):
        print("\nResolving coreferences...\n")
        text = resolve_references(text)
        write_to_file("coref_output.txt", text)
    else:
        print("\nNo coreferences to resolve\n")

    sents_data = get_sentence_data(text)

    parallel_sentences = find_sentences_with_parallel_keywords(sents_data)
    loop_sentences = find_sentences_with_loop_keywords(sents_data)

    data = get_model_outputs(text)

    if not data:
        return

    agents, tasks, conditions = extract_all_entities(data)

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents_data)

    if len(conditions) > 0:
        agent_task_pairs = add_conditions(conditions, agent_task_pairs, sents_data)

    agent_task_pairs = add_parallel(agent_task_pairs, sents_data, parallel_sentences)
    agent_task_pairs = add_task_ids(agent_task_pairs, sents_data, loop_sentences)
    agent_task_pairs = add_loops(agent_task_pairs, sents_data, loop_sentences)
    write_to_file("agent_task_pairs.txt", agent_task_pairs)

    end_of_blocks = detect_end_of_block(sents_data)

    if len(end_of_blocks) != 0:

        agent_task_pairs_before_end_of_block = [
            x
            for x in agent_task_pairs
            if "task" in x
            and x["task"]["start"] < end_of_blocks[0]
            or "start" in x
            and x["start"] < end_of_blocks[0]
        ]
        agent_task_pairs_after_end_of_block = [
            x
            for x in agent_task_pairs
            if "task" in x
            and x["task"]["start"] >= end_of_blocks[0]
            or "start" in x
            and x["start"] >= end_of_blocks[0]
        ]

        output_1 = create_bpmn_structure(agent_task_pairs_before_end_of_block)
        output_2 = create_bpmn_structure(agent_task_pairs_after_end_of_block)

        if isinstance(output_1, dict):
            output_1 = [output_1]

        if isinstance(output_2, dict):
            output_2 = [output_2]

        output = output_1 + output_2

    else:
        output = create_bpmn_structure(agent_task_pairs)

    write_to_file("final_output.txt", output)

    return output


def generate_graph(input, notebook):

    bpmn = GraphGenerator(input, notebook)
    print("Generating graph...\n")
    bpmn.generate_graph()
    bpmn.show()


if __name__ == "__main__":

    text, notebook = parse_arguments()
    delete_files_in_folder("./output_logs")
    try:
        output = process_text(text)
    except:
        print("Error when processing text")
        exit()
    try:
        generate_graph(output, notebook)
    except:
        print("Error when generating graph")
