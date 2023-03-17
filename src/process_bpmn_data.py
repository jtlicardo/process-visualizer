import json

import requests
import spacy
from sentence_transformers import SentenceTransformer, util
from spacy.matcher import Matcher

import openai_prompts as prompts
from coreference_resolution.coref import resolve_references
from graph_generator import GraphGenerator
from logging_utils import clear_folder, write_to_file


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

    write_to_file("sentence_data.txt", sentence_data)
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
    write_to_file("model_output.txt", data)

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
            "parameters": {"candidate_labels": ["process start", "process end"]},
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
        entity["entity_group"] = (
            "PROCESS_START" if data["labels"][0] == "process start" else "PROCESS_END"
        )
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


def add_conditions(conditions: list, agent_task_pairs: list, sentences: list) -> list:
    """
    Adds conditions and condition ids to agent-task pairs
    Args:
        conditions (list): a list of conditions
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): a list of sentences
    Returns:
        list: a list of agent-task pairs with conditions and condition ids
    """

    updated_agent_task_pairs = []
    condition_id = 0

    for pair in agent_task_pairs:

        task = pair["task"]

        for sent in sentences:

            if sent["start"] <= task["start"] <= sent["end"]:

                for condition in conditions:
                    if sent["start"] <= condition["start"] <= sent["end"]:
                        pair["condition"] = condition
                        pair["condition"]["condition_id"] = f"C{condition_id}"
                        condition_id += 1
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


def get_conditions(agent_task_pairs: list) -> list:
    """
    Gets the conditions from the agent-task pairs.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
    Returns:
        list: the list of conditions
    """
    return [pair["condition"] for pair in agent_task_pairs if "condition" in pair]


def check_if_conditions_in_same_gateway(
    process_description: str, conditions: list
) -> list:
    """
    Checks whether conditions belong to the same exclusive gateway.
    Args:
        process_description (str): the process description
        conditions (list): the list of conditions
    Returns:
        list: list of dictionaries, each dictionary contains the condition ids and the result of the check
    """

    result = []
    obj = {}

    for i, condition in enumerate(conditions):
        if i < len(conditions) - 1:
            condition_pair = f"'{condition['word']}' and '{conditions[i+1]['word']}'"
            obj = {
                "condition_1": condition["condition_id"],
                "condition_2": conditions[i + 1]["condition_id"],
            }
            response = prompts.same_exclusive_gateway(
                process_description, condition_pair
            )
            obj["result"] = response["content"]
            result.append(obj)

    return result


def assign_exclusive_gateway_ids(results: list) -> dict:
    """
    Assigns exclusive gateway ids to each condition.
    Args:
        results (list): the list of dictionaries, each dictionary contains the condition ids and the result of the check
    Returns:
        dict: dictionary with condition ids as keys and exclusive gateway ids as values
    """

    conditions = {}
    id = 0

    for result in results:
        if result["result"] == "TRUE":
            if result["condition_1"] not in conditions:
                conditions[result["condition_1"]] = f"EG{id}"
            conditions[result["condition_2"]] = f"EG{id}"
        else:
            if result["condition_1"] not in conditions:
                conditions[result["condition_1"]] = f"EG{id}"
            conditions[result["condition_2"]] = f"EG{id + 1}"
        id += 1

    return conditions


def add_exclusive_gateway_ids(agent_task_pairs: list, conditions: dict):
    """
    Adds exclusive gateway ids to agent-task pairs
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        conditions (dict): dictionary with condition ids as keys and exclusive gateway ids as values
    Returns:
        list: the list of agent-task pairs with exclusive gateway ids
    """

    updated_agent_task_pairs = []

    for pair in agent_task_pairs:
        if "condition" in pair:
            pair["condition"]["exclusive_gateway_id"] = conditions[
                pair["condition"]["condition_id"]
            ]
        updated_agent_task_pairs.append(pair)

    return updated_agent_task_pairs


def handle_conditions(
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
    conditions_with_ids = get_conditions(agent_task_pairs)
    result = check_if_conditions_in_same_gateway(process_desc, conditions_with_ids)
    conditions_with_exclusive_gateway_ids = assign_exclusive_gateway_ids(result)
    updated_agent_task_pairs = add_exclusive_gateway_ids(
        updated_agent_task_pairs, conditions_with_exclusive_gateway_ids
    )

    return updated_agent_task_pairs


def find_second_condition_index(dict_list: list) -> int:
    """
    Finds the second condition that has the same exclusive gateway id.
    If there is no second condition that has the same exclusive gateway id, returns -1.
    Args:
        dict_list (list): the list of dictionaries
    Returns:
        int: the index of the second condition that has the same exclusive gateway id
    """

    found_conditions = 0
    for i, d in enumerate(dict_list):
        if "condition" in d:
            if (
                d["condition"]["exclusive_gateway_id"]
                == dict_list[1]["condition"]["exclusive_gateway_id"]
            ):
                found_conditions += 1
                if found_conditions == 2:
                    return i
    return -1


def find_first_task_in_next_sentence(dict_list: list) -> tuple:
    """
    Finds the first task in the next sentence. If there is no next sentence, returns a tuple of None and None.
    Args:
        dict_list (list): the list of dictionaries
    Returns:
        tuple: the first task in the next sentence and its index
    """
    if dict_list[0]["sentence_idx"] == dict_list[-1]["sentence_idx"]:
        return (None, None)
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


def process_text(text):

    clear_folder("./output_logs")

    print("\nInput text:\n" + text)

    if should_resolve_coreferences(text):
        print("\nResolving coreferences...\n")
        text = resolve_references(text)
        write_to_file("coref_output.txt", text)
    else:
        print("\nNo coreferences to resolve\n")

    sents_data = create_sentence_data(text)

    parallel_sentences = find_sentences_with_parallel_keywords(sents_data)
    loop_sentences = find_sentences_with_loop_keywords(sents_data)

    data = extract_bpmn_data(text)

    if not data:
        return

    agents, tasks, conditions, process_info = extract_all_entities(data)

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents_data)

    if len(conditions) > 0:
        agent_task_pairs = handle_conditions(
            agent_task_pairs, conditions, sents_data, text
        )

    if len(process_info) > 0:
        process_info = batch_classify_process_info(process_info)
        agent_task_pairs = add_process_end_events(
            agent_task_pairs, sents_data, process_info
        )

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
