from logging_utils import write_to_file


def create_bpmn_structure(
    agent_task_pairs, parallel_gateway_data, exclusive_gateway_data
):

    format_agent_task_pairs(agent_task_pairs)

    agent_task_pairs_to_add = agent_task_pairs.copy()

    gateways = parallel_gateway_data + exclusive_gateway_data
    gateways = sorted(gateways, key=calculate_distance)

    add_tasks_to_gateways(agent_task_pairs_to_add, gateways)

    write_to_file("bpmn_structure/gateways.json", gateways)

    nested_gateways = nest_gateways(gateways)

    write_to_file("bpmn_structure/nested_gateways.json", nested_gateways)

    structure = agent_task_pairs_to_add + nested_gateways
    structure = sorted(structure, key=lambda x: get_start_idx(x))

    return structure


def get_start_idx(dictionary):
    if "start" in dictionary:
        return dictionary["start"]
    elif "content" in dictionary and "task" in dictionary["content"]:
        return dictionary["content"]["task"]["start"]
    else:
        return None


def format_agent_task_pairs(agent_task_pairs):

    for pair in agent_task_pairs:
        pair["content"] = pair.copy()
        pair["type"] = "task" if "task" in pair else "loop"
        for key in pair.copy():
            if key != "type" and key != "content":
                del pair[key]


def add_tasks_to_gateways(agent_task_pairs_to_add, gateways):

    for gateway in gateways:
        gateway["type"] = "parallel" if gateway["id"].startswith("PG") else "exclusive"
        gateway["children"] = [[] for _ in range(len(gateway["paths"]))]
        for i, path in enumerate(gateway["paths"]):
            for pair in agent_task_pairs_to_add:
                start_idx = (
                    pair["content"]["task"]["start"]
                    if pair["type"] == "task"
                    else pair["content"]["start"]
                )
                if start_idx in range(path["start"], path["end"] + 1):
                    gateway["children"][i].append(pair)
                    agent_task_pairs_to_add.remove(pair)

                    if "condition" in pair["content"] and gateway["id"].startswith(
                        "PG"
                    ):
                        if "condition" not in gateway:
                            gateway["condition"] = pair["content"]["condition"]
                        del pair["content"]["condition"]


def calculate_distance(gateway):
    return gateway["end"] - gateway["start"]


def nest_gateways(all_gateways):
    def is_nested(inner, outer):
        return (
            inner["start"] >= outer["start"]
            and inner["end"] <= outer["end"]
            and (inner["start"] > outer["start"] or inner["end"] < outer["end"])
        )

    def find_parent_and_path(gateway):
        parent = None
        parent_path = None
        for candidate in all_gateways:
            for path in candidate["paths"]:
                if is_nested(gateway, path):
                    if parent is None or is_nested(path, parent_path):
                        parent = candidate
                        parent_path = path
        return parent, parent_path

    def insert_in_sorted_order(children, gateway):
        index = 0
        try:
            child_start_idx = (
                children[0]["content"]["task"]["start"]
                if children[0]["type"] == "task"
                else children[0]["start"]
            )
        except IndexError:
            pass
        while index < len(children) and child_start_idx < gateway["start"]:
            index += 1
        children.insert(index, gateway)

    for gateway in all_gateways:
        parent, parent_path = find_parent_and_path(gateway)
        if parent is not None:
            path_index = parent["paths"].index(parent_path)
            insert_in_sorted_order(parent["children"][path_index], gateway)

    top_level_gateways = [
        gateway for gateway in all_gateways if find_parent_and_path(gateway)[0] is None
    ]

    return top_level_gateways


if __name__ == "__main__":
    # Used for testing
    import json
    from os.path import exists

    parallel_gateway_data = []
    exclusive_gateway_data = []

    with open("output_logs/agent_task_pairs.json", "r") as file:
        agent_task_pairs = file.read()
    agent_task_pairs = json.loads(agent_task_pairs)

    if exists("output_logs/parallel_gateway_data.json"):
        with open("output_logs/parallel_gateway_data.json", "r") as file:
            parallel_gateway_data = file.read()
        parallel_gateway_data = json.loads(parallel_gateway_data)

    if exists("output_logs/exclusive_gateway_data.json"):
        with open("output_logs/exclusive_gateway_data.json", "r") as file:
            exclusive_gateway_data = file.read()
        exclusive_gateway_data = json.loads(exclusive_gateway_data)

    structure = create_bpmn_structure(
        agent_task_pairs, parallel_gateway_data, exclusive_gateway_data
    )
    write_to_file("bpmn_structure/bpmn_structure.json", structure)
