import json

import requests
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy.matcher import Matcher

from coreference_resolution.coref import resolve_references
from graph_generator import GraphGenerator
from logging_utils import delete_files_in_folder, write_to_file

API_URL = "https://api-inference.huggingface.co/models/jtlicardo/bpmn-information-extraction-v2"


def get_sentences(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [str(i) for i in list(doc.sents)]
    return sentences


def create_sentence_data(sentences):

    counter = 0
    sentence_data = []

    for sent in sentences:
        start = counter
        end = counter + len(sent)
        counter += len(sent) + 1
        sentence = {"sentence": sent, "start": start, "end": end}
        sentence_data.append(sentence)

    return sentence_data


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, data=data)
    return json.loads(response.content.decode("utf-8"))


def extract_entities(type: str, data: list) -> list:
    """
    Extracts all entities of a given type from the model output
    Args:
        type (str): the type of entity to extract
        data (list): the model output
    Returns:
        list: a list of entities of the given type
    """
    agents = []
    for entity in data:
        if entity["entity_group"] == type and entity["score"] > 0.5:
            agents.append(entity)
    return agents


def create_agent_task_pairs(agents, tasks, sentences):

    agents_in_sentences = []
    tasks_in_sentences = []
    agent_task_pairs = []

    for agent in agents:
        for i, sent in enumerate(sentences):
            if sent["start"] <= agent["start"] <= sent["end"]:
                agents_in_sentences.append({"index": i, "agent": agent})

    for task in tasks:
        for i, sent in enumerate(sentences):
            if sent["start"] <= task["start"] <= sent["end"]:
                tasks_in_sentences.append({"index": i, "task": task})

    for agent in agents_in_sentences:
        for task in tasks_in_sentences:
            if agent["index"] == task["index"]:
                agent_task_pairs.append(
                    {
                        "agent": agent["agent"],
                        "task": task["task"],
                        "sentence_idx": task["index"],
                    }
                )

    return agent_task_pairs


def add_conditions(conditions, agent_task_pairs, sentences):

    updated_agent_task_pairs = []

    for pair in agent_task_pairs:

        task = pair["task"]

        for sent in sentences:

            if sent["start"] <= task["start"] <= sent["end"]:

                for condition in conditions:
                    if sent["start"] <= condition["start"] <= sent["end"]:
                        pair["condition"] = condition
                        break

        updated_agent_task_pairs.append(pair)

    return updated_agent_task_pairs


def detect_end_of_block(sentences):

    nlp = spacy.load("en_core_web_md")

    matcher = Matcher(nlp.vocab)

    pattern_1 = [
        {"LOWER": "after", "IS_SENT_START": True},
        {"LOWER": "that", "OP": "!"},
    ]
    pattern_2 = [{"LOWER": "once", "IS_SENT_START": True}]
    pattern_3 = [{"LOWER": "when", "IS_SENT_START": True}]
    pattern_4 = [{"LOWER": "upon", "IS_SENT_START": True}]
    pattern_5 = [{"LOWER": "concluding", "IS_SENT_START": True}]
    matcher.add("END_OF_BLOCK", [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5])

    end_of_blocks = []

    for sent in sentences:

        doc = nlp(sent["sentence"])

        matches = matcher(doc)

        if len(matches) > 0:
            end_of_blocks.append(sent["start"])

    return end_of_blocks


def find_sentences_with_parallel_keywords(sentences):

    nlp = spacy.load("en_core_web_md")

    matcher = Matcher(nlp.vocab)

    pattern_1 = [{"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "meantime"}]
    pattern_2 = [
        {"LOWER": "at"},
        {"LOWER": "the"},
        {"LOWER": "same"},
        {"LOWER": "time"},
    ]
    pattern_3 = [{"LOWER": "meanwhile"}]
    pattern_4 = [{"LOWER": "while"}]
    pattern_5 = [{"LOWER": "in"}, {"LOWER": "parallel"}]
    pattern_6 = [{"LOWER": "concurrently"}]

    matcher.add(
        "PARALLEL", [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5, pattern_6]
    )

    detected_sentences = []

    for sent in sentences:

        doc = nlp(sent["sentence"])

        matches = matcher(doc)

        if len(matches) > 0:
            detected_sentences.append(sent)

    return detected_sentences


def find_sentences_with_loop_keywords(sentences):

    nlp = spacy.load("en_core_web_md")

    matcher = Matcher(nlp.vocab)

    pattern_1 = [{"LOWER": "again"}]

    matcher.add("LOOP", [pattern_1])

    detected_sentences = []

    for sent in sentences:

        doc = nlp(sent["sentence"])

        matches = matcher(doc)

        if len(matches) > 0:
            detected_sentences.append(sent)

    return detected_sentences


def add_parallel(agent_task_pairs, sentences, parallel_sentences):

    updated_agent_task_pairs = []

    for pair in agent_task_pairs:

        task = pair["task"]

        for sent in sentences:

            if sent["start"] <= task["start"] <= sent["end"]:

                if sent in parallel_sentences:
                    pair["in_sentence_with_parallel_keyword"] = True
                else:
                    pair["in_sentence_with_parallel_keyword"] = False

        updated_agent_task_pairs.append(pair)

    return updated_agent_task_pairs


def compare_tasks(task1: str, task2: str) -> float:
    """
    Compares two tasks using the sentence-transformers model
    Args:
        task1 (str): the first task
        task2 (str): the second task
    Returns:
        float: the cosine similarity score
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings1 = model.encode(task1, convert_to_tensor=True)
    embeddings2 = model.encode(task2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores[0][0]


def find_task_with_highest_similarity(task: dict, list: list) -> str:
    """
    Finds the task in the list that has the highest cosine similarity with the given task
    Args:
        task (dict): the task to compare
        list (list): the list of tasks to compare with
    Returns:
        str: the task with the highest cosine similarity
    """
    highest_cosine_similarity = 0
    highest_cosine_similarity_task = None
    for t in list:
        cosine_similarity = compare_tasks(task["word"], t["word"])
        if cosine_similarity > highest_cosine_similarity:
            highest_cosine_similarity = cosine_similarity
            highest_cosine_similarity_task = t
    return highest_cosine_similarity_task


def num_of_tasks_in_sentence(agent_task_pairs: list, sentence_idx: int) -> int:
    """
    Counts the number of tasks in a sentence
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        sentence_idx (int): the index of the sentence
    Returns:
        int: the number of tasks in the sentence
    """
    num_of_tasks = 0
    for pair in agent_task_pairs:
        if pair["sentence_idx"] == sentence_idx:
            num_of_tasks += 1
    return num_of_tasks


def add_task_ids(agent_task_pairs, sentences, loop_sentences):
    """
    Adds task ids to tasks that are not in a sentence with a loop keyword.
    """

    updated_agent_task_pairs = []
    id = 0

    for pair in agent_task_pairs:
        task = pair["task"]
        for sent in sentences:
            if sent["start"] <= task["start"] <= sent["end"]:
                if sent not in loop_sentences:
                    pair["task"]["task_id"] = f"T{id}"
                    id += 1

        updated_agent_task_pairs.append(pair)

    return updated_agent_task_pairs


def add_loops(agent_task_pairs, sentences, loop_sentences):
    """
    Adds go_to fields to tasks that are in a sentence with a loop keyword.
    """
    updated_agent_task_pairs = []
    previous_tasks = []

    for pair in agent_task_pairs:

        task = pair["task"]

        for sent in sentences:

            if sent["start"] <= task["start"] <= sent["end"]:

                if sent in loop_sentences:
                    highest_similarity_task = find_task_with_highest_similarity(
                        task, previous_tasks
                    )
                    if highest_similarity_task is not None:
                        pair["go_to"] = highest_similarity_task["task_id"]
                        pair["start"] = pair["task"]["start"]
                        del pair["task"]
                        del pair["agent"]
                else:
                    previous_tasks.append(task)

        updated_agent_task_pairs.append(pair)

    return updated_agent_task_pairs


def find_second_condition_index(dict_list):
    found_conditions = 0
    for i, d in enumerate(dict_list):
        if "condition" in d:
            found_conditions += 1
            if found_conditions == 2:
                return i
    return -1


def find_first_task_in_next_sentence(dict_list: list) -> tuple:
    """
    Finds the first task in the next sentence
    Args:
        dict_list (list): the list of dictionaries
    Returns:
        tuple: the first task in the next sentence and its index
    """
    if dict_list[0]["sentence_idx"] == dict_list[-1]["sentence_idx"]:
        return None, None
    cur_sentence_idx = dict_list[0]["sentence_idx"]
    next_sentence_idx = cur_sentence_idx + 1
    for i, x in enumerate(dict_list):
        if x["sentence_idx"] == next_sentence_idx:
            return (x, i)


def create_bpmn_structure(input):

    if len(input) == 1:
        type = "task" if ("task" in input[0]) else "loop"
        return {"type": type, "content": input[0]}
    elif isinstance(input, dict):
        type = "task" if ("task" in input) else "loop"
        return {"type": type, "content": input}

    first_task, idx = find_first_task_in_next_sentence(input)

    if first_task and idx:
        if first_task["in_sentence_with_parallel_keyword"] == True:
            if "condition" in input[0]:
                value = input[0].pop("condition")
                return [
                    {
                        "type": "parallel",
                        "condition": value,
                        "children": [
                            create_bpmn_structure(input[0:idx]),
                            create_bpmn_structure(input[idx:]),
                        ],
                    }
                ]
            else:
                return [
                    {
                        "type": "parallel",
                        "children": [
                            create_bpmn_structure(input[0:idx]),
                            create_bpmn_structure(input[idx:]),
                        ],
                    }
                ]
        else:
            if "condition" in input[1]:
                # print(input)
                second_condition_idx = find_second_condition_index(input)

                if second_condition_idx == -1:
                    return [
                        {"type": "task", "content": input[0]},
                        {
                            "type": "exclusive",
                            "children": create_bpmn_structure(input[1:]),
                        },
                    ]
                else:
                    return [
                        {"type": "task", "content": input[0]},
                        {
                            "type": "exclusive",
                            "children": [
                                create_bpmn_structure(input[1:second_condition_idx]),
                                create_bpmn_structure(input[second_condition_idx:]),
                            ],
                        },
                    ]
            else:
                remainder = create_bpmn_structure(input[1:])
                if isinstance(remainder, list):
                    return [create_bpmn_structure(input[0])] + [*remainder]
                else:
                    return [create_bpmn_structure(input[0])] + [remainder]
    else:
        remainder = create_bpmn_structure(input[1:])
        if isinstance(remainder, list):
            return [create_bpmn_structure(input[0])] + [*remainder]
        else:
            return [create_bpmn_structure(input[0])] + [remainder]


def should_resolve_coreferences(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if token.lower_ in ["he", "she", "it", "they"]:
            return True
    return False

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

    delete_files_in_folder("./output_logs")

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

def generate_graph_pdf(input, notebook):

    bpmn = GraphGenerator(input, notebook=notebook)
    print("Generating graph...\n")
    bpmn.generate_graph()
    bpmn.show()

def generate_graph_image(input):
    bpmn = GraphGenerator(input, format="jpeg", notebook=False)
    print("Generating graph...\n")
    bpmn.generate_graph()
    bpmn.save_file()