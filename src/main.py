from process_bpmn_data import *
from logging_utils import write_to_file
from graph_generator import GraphGenerator
import argparse
from coreference_resolution.coref import resolve_references


def generate_graph():

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

    print("\nInput text:\n" + text)

    if should_resolve_coreferences(text):
        print("\nResolving coreferences...\n")
        text = resolve_references(text)
        write_to_file("coref_output.txt", text)
    else:
        print("\nNo coreferences to resolve\n")

    sents = get_sentences(text)
    sents_data = create_sentence_data(sents)
    write_to_file("sentences.txt", sents_data)

    parallel_sentences = find_sentences_with_parallel_keywords(sents_data)

    print("Getting model outputs...\n")
    data = query({"inputs": text, "options": {"wait_for_model": True}})
    write_to_file("model_output.txt", data)

    if "error" in data:
        print("Error when getting model outputs:", data["error"])
        return

    print("Extracting entities...\n")
    agents = extract_entities("AGENT", data)
    tasks = extract_entities("TASK", data)
    conditions = extract_entities("CONDITION", data)

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents_data)
    write_to_file("agent_task_pairs_0.txt", agent_task_pairs)

    if len(conditions) > 0:
        agent_task_pairs = add_conditions(conditions, agent_task_pairs, sents_data)

    agent_task_pairs = add_parallel(agent_task_pairs, sents_data, parallel_sentences)

    end_of_blocks = detect_end_of_block(sents_data)

    if len(end_of_blocks) != 0:

        agent_task_pairs_before_end_of_block = [
            x for x in agent_task_pairs if x["task"]["start"] < end_of_blocks[0]
        ]
        agent_task_pairs_after_end_of_block = [
            x for x in agent_task_pairs if x["task"]["start"] >= end_of_blocks[0]
        ]

        output_1 = create_bpmn_structure(agent_task_pairs_before_end_of_block)
        output_2 = create_bpmn_structure(agent_task_pairs_after_end_of_block)

        if isinstance(output_1, dict):
            output_1 = [output_1]

        if isinstance(output_2, dict):
            output_2 = [output_2]

        output = output_1 + output_2

    else:
        write_to_file("agent_task_pairs_1.txt", agent_task_pairs)
        output = create_bpmn_structure(agent_task_pairs)

    write_to_file("final_output.txt", output)

    if args.notebook:
        bpmn = GraphGenerator(output, notebook=True)
    else:
        bpmn = GraphGenerator(output)

    print("Generating graph...\n")
    bpmn.generate_graph()
    bpmn.show()


if __name__ == "__main__":
    generate_graph()
