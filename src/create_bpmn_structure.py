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

    write_to_file("bpmn_structure/bpmn_structure.json", structure)

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
            for pair in agent_task_pairs_to_add.copy():
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
        if gateway["type"] == "exclusive":
            # Only the first agent pair in the children list can contain the "condition" key
            for i in range(len(gateway["children"])):
                children_list = gateway["children"][i]
                for j in range(len(children_list)):
                    child = children_list[j]
                    if (
                        j > 0
                        and (child["type"] == "task" or child["type"] == "loop")
                        and "condition" in child["content"]
                    ):
                        del child["content"]["condition"]


def calculate_distance(gateway):
    return gateway["end"] - gateway["start"]


def nest_gateways(all_gateways):
    def is_nested(inner, outer):
        if (
            inner["start"] >= outer["start"]
            and inner["end"] <= outer["end"]
            and (inner["start"] > outer["start"] or inner["end"] < outer["end"])
        ):
            return True
        else:
            range_1 = (inner["start"], inner["end"])
            range_2 = (outer["start"], outer["end"])
            return ranges_overlap_percentage(range_1, range_2)

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
        child_start_idx = (
            children[0]["content"]["task"]["start"]
            if children[0]["type"] == "task"
            else children[0]["start"]
        )
        while index < len(children) and child_start_idx < gateway["start"]:
            index += 1
            child_start_idx = (
                children[index]["content"]["task"]["start"]
                if children[index]["type"] == "task"
                else children[index]["start"]
            )
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


def ranges_overlap_percentage(range1, range2, min_overlap_percentage=0.97):
    start1, end1 = range1
    start2, end2 = range2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start < overlap_end:
        overlap_range = overlap_end - overlap_start
        range1_size = end1 - start1
        range2_size = end2 - start2

        overlap_percentage1 = overlap_range / range1_size
        overlap_percentage2 = overlap_range / range2_size

        return (
            overlap_percentage1 >= min_overlap_percentage
            and overlap_percentage2 >= min_overlap_percentage
        )
    else:
        return False


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
