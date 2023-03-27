import json
import re

import requests
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy.matcher import Matcher
from thefuzz import fuzz

import openai_prompts as prompts
from coreference_resolution.coref import resolve_references
from graph_generator import GraphGenerator
from logging_utils import clear_folder, write_to_file
from create_bpmn_structure import create_bpmn_structure


BPMN_INFORMATION_EXTRACTION_ENDPOINT = "https://api-inference.huggingface.co/models/jtlicardo/bpmn-information-extraction-v2"
ZERO_SHOT_CLASSIFICATION_ENDPOINT = (
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
)


def get_sentences(text: str) -> list:
    """
    Creates a list of sentences from a given text.
    Args:
        text (str): the text to split into sentences
    Returns:
        list: a list of sentences
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [str(i) for i in list(doc.sents)]
    return sentences


def create_sentence_data(text: str) -> list:
    """
    Creates a list of dictionaries containing the sentence data (sentence, start index, end index)
    Args:
        text (str): the input text
    Returns:
        list: a list of dictionaries containing the sentence data
    """

    sentences = get_sentences(text)

    start = 0
    sentence_data = []

    for sent in sentences:
        end = start + len(sent)
        sentence_data.append({"sentence": sent, "start": start, "end": end})
        start += len(sent) + 1

    write_to_file("sentence_data.json", sentence_data)
    return sentence_data


def query(payload, endpoint):
    data = json.dumps(payload)
    response = requests.request("POST", endpoint, data=data)
    return json.loads(response.content.decode("utf-8"))


def extract_bpmn_data(text: str) -> list:
    """
    Extracts BPMN data from the process description by calling the model endpoint hosted on Hugging Face.
    Args:
        text (str): the process description
    Returns:
        list: model output
    """

    print("Extracting BPMN data...\n")
    data = query(
        {"inputs": text, "options": {"wait_for_model": True}},
        BPMN_INFORMATION_EXTRACTION_ENDPOINT,
    )
    write_to_file("model_output.json", data)

    if "error" in data:
        print("Error when extracting BPMN data:", data["error"])
        return None

    return data


def classify_process_info(text: str) -> dict:
    """
    Classifies a PROCESS_INFO entity by calling the model endpoint hosted on Hugging Face.
    Args:
        text (str): sequence of text classified as process info
    Returns:
        dict: model output containing the following keys: "sequence", "labels", "scores"
    """

    data = query(
        {
            "inputs": text,
            "parameters": {"candidate_labels": ["start", "end", "split", "return"]},
            "options": {"wait_for_model": True},
        },
        ZERO_SHOT_CLASSIFICATION_ENDPOINT,
    )

    if "error" in data:
        print("Error when classifying PROCESS_INFO entity:", data["error"])
        return None

    return data


def batch_classify_process_info(process_info_entities: list):
    """
    Classifies a list of PROCESS_INFO entities into PROCESS_START or PROCESS_END.
    Args:
        process_info_entities (list): a list of PROCESS_INFO entities
    Returns:
        list: a list of PROCESS_INFO entities with the entity_group key updated
    """

    updated_entities = []

    print("Classifying PROCESS_INFO entities...\n")

    for entity in process_info_entities:
        text = entity["word"]
        data = classify_process_info(text)
        process_info_dict = {
            "start": "PROCESS_START",
            "end": "PROCESS_END",
            "split": "PROCESS_SPLIT",
            "return": "PROCESS_RETURN",
        }
        entity["entity_group"] = process_info_dict[data["labels"][0]]
        updated_entities.append(entity)

    return updated_entities


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
    """
    Combines agents and tasks into agent-task pairs based on the sentence they appear in.
    Args:
        agents (list): a list of agents
        tasks (list): a list of tasks
        sentences (list): a list of sentence data
    Returns:
        list: a list of agent-task pairs (each pair is a dictionary containing the agent, task and sentence index)
    """

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

    multi_agent_sentences_idx = []
    multi_agent_sentences = []

    # Check if multiple agents appear in the same sentence
    for i, agent in enumerate(agents_in_sentences):
        if i != len(agents_in_sentences) - 1:
            if agent["index"] == agents_in_sentences[i + 1]["index"]:
                print(
                    "Multiple agents in sentence:",
                    sentences[agent["index"]]["sentence"],
                    "\n",
                )
                multi_agent_sentences_idx.append(agent["index"])

    # For sentences that contain multiple agents, connect agents and tasks based on their order in the sentence
    # For example, if the sentence is "A does B and C does D", then the agent-task pairs are (A, B) and (C, D)

    for idx in multi_agent_sentences_idx:
        sentence_data = {
            "sentence_idx": idx,
            "agents": [agent for agent in agents_in_sentences if agent["index"] == idx],
            "tasks": [task for task in tasks_in_sentences if task["index"] == idx],
        }
        multi_agent_sentences.append(sentence_data)

    for sentence in multi_agent_sentences:
        for i, agent in enumerate(sentence["agents"]):
            agent_task_pairs.append(
                {
                    "agent": agent["agent"],
                    "task": sentence["tasks"][i]["task"],
                    "sentence_idx": sentence["sentence_idx"],
                }
            )

    for agent in agents_in_sentences:
        for task in tasks_in_sentences:
            if (
                agent["index"] == task["index"]
                and agent["index"] not in multi_agent_sentences_idx
            ):
                agent_task_pairs.append(
                    {
                        "agent": agent["agent"],
                        "task": task["task"],
                        "sentence_idx": task["index"],
                    }
                )

    agent_task_pairs = sorted(agent_task_pairs, key=lambda k: k["sentence_idx"])

    return agent_task_pairs


def add_process_end_events(
    agent_task_pairs: list, sentences: list, process_info_entities: list
) -> list:
    """
    Adds process end events to agent-task pairs
    Args:
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): list of sentence data
        process_info_entities (list): list of process info entities
    Returns:
        list: a list of agent-task pairs with process end events
    """

    process_end_events = []

    for entity in process_info_entities:
        if entity["entity_group"] == "PROCESS_END":
            for sent in sentences:
                if sent["start"] <= entity["start"] <= sent["end"]:
                    process_end_events.append(
                        {
                            "process_end_event": entity,
                            "sentence_idx": sentences.index(sent),
                        }
                    )

    for pair in agent_task_pairs:
        for event in process_end_events:
            if pair["sentence_idx"] == event["sentence_idx"]:
                pair["process_end_event"] = event["process_end_event"]

    return agent_task_pairs


def has_parallel_keywords(text: str) -> bool:
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
    pattern_7 = [{"LOWER": "simultaneously"}]

    matcher.add(
        "PARALLEL",
        [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5, pattern_6, pattern_7],
    )

    doc = nlp(text)

    matches = matcher(doc)

    if len(matches) > 0:
        return True

    return False


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


def add_exclusive_gateway_ids(agent_task_pairs: list, conditions: dict) -> list:
    """
    Adds exclusive gateway ids to agent-task pairs.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        conditions (dict): the key is the condition and the value is the exclusive gateway id
    Returns:
        list: the list of agent-task pairs with exclusive gateway ids
    """

    # Adding the exclusive gateway ids to the agent-task pairs that have a condition
    for pair in agent_task_pairs:
        if "condition" in pair:
            condition = pair["condition"]["word"]
            max_prob_gateway = ""
            max_prob = 0
            for key, value in conditions.items():
                prob = fuzz.ratio(condition, key)
                if prob > max_prob:
                    max_prob = prob
                    max_prob_gateway = value
            assert max_prob_gateway != "", "No exclusive gateway id found"
            pair["exclusive_gateway_id"] = max_prob_gateway

    return agent_task_pairs


def extract_exclusive_gateways(process_description: str, conditions: list) -> list:
    """
    Extracts the conditions for each exclusive gateway in the process description
    Args:
        process_description (str): the process description
        conditions (list): the list of condition entities found in the process description
    Returns:
        list: the list of exclusive gateways

    Example output:
    [
        {
            "id": "EG0",
            "conditions": [
                "if the customer is a new customer",
                "if the customer is an existing customer"
            ],
            "start": 54,
            "end": 251,
            "paths": [{'start': 54, 'end': 210}, {'start': 211, 'end': 251}]
        },
    ]
    """

    condition_string = ""
    for condition in conditions:
        condition_text = condition["word"]
        if condition_string == "":
            condition_string += f"'{condition_text}'"
        else:
            condition_string += f", '{condition_text}'"

    response = prompts.extract_gateway_conditions(process_description, condition_string)

    pattern = r"Exclusive gateway (\d+): (.+)"
    matches = re.findall(pattern, response)
    gateway_conditions = [match[1] for match in matches]
    exclusive_gateways = [
        {"id": f"EG{i}", "conditions": [x.strip() for x in gateway.split("||")]}
        for i, gateway in enumerate(gateway_conditions)
    ]

    for gateway in exclusive_gateways:
        indices = get_indices(gateway["conditions"], process_description)
        gateway["start"] = indices[0]["start"]
        gateway["end"] = indices[-1]["end"]
        gateway["paths"] = indices

    for exclusive_gateway in exclusive_gateways:
        for i, path in enumerate(exclusive_gateway["paths"]):
            if i != len(exclusive_gateway["paths"]) - 1:
                path["end"] = exclusive_gateway["paths"][i + 1]["start"] - 1

    print("Exclusive gateways data:", exclusive_gateways, "\n")
    return exclusive_gateways


def add_conditions(conditions: list, agent_task_pairs: list, sentences: list) -> list:
    """
    Adds conditions and condition ids to agent-task pairs
    Args:
        conditions (list): a list of conditions
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): a list of sentences
    Returns:
        list: a list of agent-task pairs with conditions
    """

    condition_id = 0

    for pair in agent_task_pairs:

        for sent in sentences:

            if pair["task"]["start"] in range(sent["start"], sent["end"]):

                for condition in conditions:
                    if condition["start"] in range(sent["start"], sent["end"]):
                        pair["condition"] = condition
                        pair["condition"]["condition_id"] = f"C{condition_id}"
                        condition_id += 1
                        break

    return agent_task_pairs


def handle_text_with_conditions(
    agent_task_pairs: list, conditions: list, sents_data: list, process_desc: str
) -> list:
    """
    Adds conditions and exclusive gateway ids to agent-task pairs.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        conditions (list): the list of conditions
        sents_data (list): the sentence data
        process_desc (str): the process description
    Returns:
        list: the list of agent-task pairs with conditions and exclusive gateway ids
    """

    updated_agent_task_pairs = add_conditions(conditions, agent_task_pairs, sents_data)

    exclusive_gateway_data = extract_exclusive_gateways(process_desc, conditions)
    conditions_with_exclusive_gateway_ids = {
        condition: d["id"]
        for d in exclusive_gateway_data
        for condition in d["conditions"]
    }

    updated_agent_task_pairs = add_exclusive_gateway_ids(
        updated_agent_task_pairs,
        conditions_with_exclusive_gateway_ids,
    )

    return updated_agent_task_pairs, exclusive_gateway_data


def should_resolve_coreferences(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if token.lower_ in ["he", "she", "it", "they"]:
            return True
    return False


def extract_all_entities(data: list) -> tuple:
    """
    Extracts all entities from the model output.
    Args:
        data (list): model output
    Returns:
        tuple: a tuple of lists containing the extracted entities
    """

    print("Extracting entities...\n")
    agents = extract_entities("AGENT", data)
    tasks = extract_entities("TASK", data)
    conditions = extract_entities("CONDITION", data)
    process_info = extract_entities("PROCESS_INFO", data)
    return (agents, tasks, conditions, process_info)


def get_indices(strings: list, text: str) -> list:
    """
    Gets the start and end indices of the given strings in the text by using fuzzy string matching.
    Args:
        strings (list): the list of strings to be found in the text
        text (str): the text in which the strings are to be found
    Returns:
        list: the list of start and end indices
    """

    indices = []
    matches = []

    for str in strings:
        potential_matches = []
        length = len(str)
        first_word = str.split()[0].lower()
        first_word_indices = [
            i for i in range(len(text)) if text.lower().startswith(first_word, i)
        ]
        for idx in first_word_indices:
            potential_match = text[idx : idx + length + 1]
            prob = fuzz.ratio(str, potential_match)
            potential_matches.append(
                {"potential_match": potential_match, "probability": prob}
            )
        matches.append(max(potential_matches, key=lambda x: x["probability"]))

    for match in matches:
        txt = match["potential_match"]
        indices.append(
            {
                "start": text.find(txt),
                "end": text.find(txt) + len(txt),
            }
        )

    return indices


def get_parallel_paths(parallel_gateway, process_description):
    num = int(prompts.number_of_parallel_paths(parallel_gateway))
    assert num <= 3, "The maximum number of parallel paths is 3"
    paths = ""
    if num == 1:
        return None
    elif num == 2:
        paths = prompts.extract_2_parallel_paths(parallel_gateway)
    elif num == 3:
        paths = prompts.extract_3_parallel_paths(parallel_gateway)
    paths = paths.split("&&")
    paths = [s.strip() for s in paths]
    indices = get_indices(paths, process_description)
    print("Parallel path indices:", indices, "\n")
    return indices


def get_parallel_gateways(text):
    response = prompts.extract_parallel_gateways(text)
    pattern = r"Parallel gateway (\d+): (.+)"
    matches = re.findall(pattern, response)
    gateways = [match[1] for match in matches]
    indices = get_indices(gateways, text)
    print("Parallel gateway indices:", indices, "\n")
    return indices


def handle_text_with_parallel_keywords(agent_task_pairs, process_description):
    """
    Extracts parallel gateways and paths from the process description.
    Args:
        process_description (str): the process description
    Returns:
        list: the list of parallel gateways

    Example output:
    [
        {
            "id": "PG0",
            "start": 0,
            "end": 100,
            "paths": [
                {
                    "start": 0,
                    "end": 50,
                },
                {
                    "start": 50,
                    "end": 100,
                },
            ],
        },
    ]
    """

    parallel_gateways = []
    parallel_gateway_id = 0

    gateway_indices = get_parallel_gateways(process_description)

    for indices in gateway_indices:
        gateway = {
            "id": f"PG{parallel_gateway_id}",
            "start": indices["start"],
            "end": indices["end"],
        }
        gateway_text = process_description[indices["start"] : indices["end"]]
        gateway["paths"] = get_parallel_paths(gateway_text, process_description)
        parallel_gateway_id += 1
        parallel_gateways.append(gateway)

    for gateway in parallel_gateways:
        for path in gateway["paths"]:
            path_text = process_description[path["start"] : path["end"]]
            if has_parallel_keywords(path_text):
                print("Parallel keywords detected in path:", path_text, "\n")
                indices = get_parallel_paths(path_text, process_description)
                gateway = {
                    "id": f"PG{parallel_gateway_id}",
                    "start": path["start"],
                    "end": path["end"],
                    "paths": indices,
                    "parent": gateway["id"],
                }
                parallel_gateway_id += 1
                parallel_gateways.append(gateway)

    print("Parallel gateway data:", parallel_gateways, "\n")

    for pair in agent_task_pairs:
        for gateway in parallel_gateways:
            if (
                pair["task"]["start"] >= gateway["start"]
                and pair["task"]["end"] <= gateway["end"]
            ):
                if "parent" not in gateway:
                    pair["parallel_gateway"] = gateway["id"]
                else:
                    pair["parent_gateway"] = gateway["parent"]
                    pair["parallel_gateway"] = gateway["id"]
            for path in gateway["paths"]:
                if (
                    pair["task"]["start"] >= path["start"]
                    and pair["task"]["end"] <= path["end"]
                ):
                    if "parent" not in gateway:
                        pair["parallel_path"] = gateway["paths"].index(path)
                    else:
                        pair["parent_path"] = pair["parallel_path"]
                        pair["parallel_path"] = gateway["paths"].index(path)

    return agent_task_pairs, parallel_gateways


def process_text(text):

    clear_folder("./output_logs")

    print("\nInput text:\n" + text + "\n")

    if should_resolve_coreferences(text):
        print("Resolving coreferences...\n")
        text = resolve_references(text)
        write_to_file("coref_output.txt", text)
    else:
        print("No coreferences to resolve\n")

    data = extract_bpmn_data(text)

    if not data:
        return

    agents, tasks, conditions, process_info = extract_all_entities(data)
    parallel_gateway_data = None
    exclusive_gateway_data = None

    sents_data = create_sentence_data(text)

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents_data)

    if has_parallel_keywords(text):
        agent_task_pairs, parallel_gateway_data = handle_text_with_parallel_keywords(
            agent_task_pairs, text
        )
        write_to_file("parallel_gateway_data.json", parallel_gateway_data)

    if len(conditions) > 0:
        agent_task_pairs, exclusive_gateway_data = handle_text_with_conditions(
            agent_task_pairs, conditions, sents_data, text
        )
        write_to_file("exclusive_gateway_data.json", exclusive_gateway_data)

    if len(process_info) > 0:
        process_info = batch_classify_process_info(process_info)
        agent_task_pairs = add_process_end_events(
            agent_task_pairs, sents_data, process_info
        )

    loop_sentences = find_sentences_with_loop_keywords(sents_data)
    agent_task_pairs = add_task_ids(agent_task_pairs, sents_data, loop_sentences)
    agent_task_pairs = add_loops(agent_task_pairs, sents_data, loop_sentences)

    write_to_file("agent_task_pairs.json", agent_task_pairs)

    structure = create_bpmn_structure(
        agent_task_pairs, parallel_gateway_data, exclusive_gateway_data
    )

    write_to_file("bpmn_structure.json", structure)

    return structure


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
