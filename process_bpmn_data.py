import json
import requests
import nltk
import spacy
from spacy.matcher import Matcher


API_URL = (
    "https://api-inference.huggingface.co/models/jtlicardo/bpmn-information-extraction"
)


def split_into_sentences(text):

    sents = nltk.sent_tokenize(text)
    sents_data = []
    counter = 0

    for sent in sents:
        start = counter
        end = counter + len(sent)
        counter += len(sent) + 1
        sentence = {"sentence": sent, "start": start, "end": end}
        sents_data.append(sentence)

    return sents_data


def split_into_sents(text):
    try:
        sents = split_into_sentences(text)
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        sents = split_into_sentences(text)
    return sents


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, data=data)
    return json.loads(response.content.decode("utf-8"))


def extract_entities(type, data):
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


def connect_conditions_with_task(conditions, agent_task_pairs, sentences):

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


def add_parallel_to_task_pairs(agent_task_pairs, sentences, parallel_sentences):

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


def find_second_condition_index(dict_list):
    found_conditions = 0
    for i, d in enumerate(dict_list):
        if "condition" in d:
            found_conditions += 1
            if found_conditions == 2:
                return i
    return -1


def find_first_task_in_next_sentence(dict_list):
    if dict_list[0]["sentence_idx"] == dict_list[-1]["sentence_idx"]:
        return None, None
    cur_sentence_idx = dict_list[0]["sentence_idx"]
    next_sentence_idx = cur_sentence_idx + 1
    for i, x in enumerate(dict_list):
        if x["sentence_idx"] == next_sentence_idx:
            return (x, i)


def create_bpmn_structure(input):

    if len(input) == 1:
        return {"type": "task", "content": input[0]}
    elif isinstance(input, dict):
        return {"type": "task", "content": input}

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

                idx = find_second_condition_index(input)

                if idx == -1:
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
                                create_bpmn_structure(input[1:idx]),
                                create_bpmn_structure(input[idx:]),
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
