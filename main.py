from process_bpmn_data import *
from graph_generator import GraphGenerator
import argparse


def generate_graph():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--text", help="Textual description of a process", required=True
    )
    parser.add_argument(
        "-n",
        "--notebook",
        help="If true, will output a .gv file that can be rendered in a notebook",
        action="store_true",
    )

    args = parser.parse_args()

    print("\nGenerating graph...\n")

    sents = split_into_sents(args.text)

    parallel_sentences = find_sentences_with_parallel_keywords(sents)

    data = query({"inputs": args.text, "options": {"wait_for_model": True}})

    agents = extract_entities("AGENT", data)
    tasks = extract_entities("TASK", data)
    conditions = extract_entities("CONDITION", data)

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents)

    if len(conditions) > 0:
        agent_task_pairs = connect_conditions_with_task(
            conditions, agent_task_pairs, sents
        )

    agent_task_pairs = add_parallel_to_task_pairs(
        agent_task_pairs, sents, parallel_sentences
    )

    end_of_blocks = detect_end_of_block(sents)

    if len(end_of_blocks) != 0:

        agent_task_pairs_before_end_of_block = [
            x for x in agent_task_pairs if x["task"]["start"] < end_of_blocks[0]
        ]
        agent_task_pairs_after_end_of_block = [
            x for x in agent_task_pairs if x["task"]["start"] >= end_of_blocks[0]
        ]

        output_1 = create_bpmn_structure(agent_task_pairs_before_end_of_block)
        output_2 = create_bpmn_structure(agent_task_pairs_after_end_of_block)

        if isinstance(output_2, dict):
            output_2 = [output_2]

        combined = output_1 + output_2

        if args.notebook:
            bpmn = GraphGenerator(combined, notebook=True)
        else:
            bpmn = GraphGenerator(combined)

    else:
        output = create_bpmn_structure(agent_task_pairs)

        if args.notebook:
            bpmn = GraphGenerator(output, notebook=True)
        else:
            bpmn = GraphGenerator(output)

    bpmn.generate_graph()
    bpmn.show()


if __name__ == "__main__":
    generate_graph()
