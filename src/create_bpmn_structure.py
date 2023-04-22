from logging_utils import write_to_file


def create_bpmn_structure(
    agent_task_pairs: list[dict],
    parallel_gateway_data: list[dict],
    exclusive_gateway_data: list[dict],
    process_info: list[dict],
) -> list[dict]:
    """
    Creates a BPMN structure from the agent-task pairs, parallel gateways and exclusive gateways.
    The BPMN structure can be used to create a visual representation of the BPMN diagram.
    Args:
        agent_task_pairs (list[dict]): A list of agent-task pairs.
        parallel_gateway_data (list[dict]): A list of parallel gateway data.
        exclusive_gateway_data (list[dict]): A list of exclusive gateway data.
        process_info (list[dict]): A list of process info entities.
    Returns:
        list[dict]: A list of BPMN structure elements.
    """

    format_agent_task_pairs(agent_task_pairs)

    agent_task_pairs_to_add = agent_task_pairs.copy()

    gateways = parallel_gateway_data + exclusive_gateway_data
    gateways = sorted(gateways, key=calculate_distance)

    add_tasks_to_gateways(agent_task_pairs_to_add, gateways, process_info)

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


def gateway_contains_nested_gateways(gateway, all_gateways):
    for g in all_gateways:
        if g["start"] > gateway["start"] and g["end"] < gateway["end"]:
            return True
    return False


def add_tasks_to_gateways(
    agent_task_pairs_to_add: list[dict], gateways: list[dict], process_info: list[dict]
) -> None:
    """
    Adds the agent-task pairs to the corresponding gateways based on their start and end indices.
    Args:
        agent_task_pairs_to_add (list[dict]): A list of agent-task pairs to add to the gateways.
        gateways (list[dict]): A list of gateways.
        process_info (list[dict]): A list of process info entities.
    Returns:
        None
    """

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
            # Handle PROCESS_CONTINUE entity in exclusive gateways
            process_continue_entities = [
                e
                for e in process_info
                if e["entity_group"] == "PROCESS_CONTINUE"
                and e["start"] in range(gateway["start"], gateway["end"] + 1)
            ]
            if len(process_continue_entities) == 1:
                handle_process_continue_entity(
                    agent_task_pairs_to_add, gateways, gateway
                )


def handle_process_continue_entity(
    agent_task_pairs_to_add: list[dict], gateways: list[dict], gateway: dict
) -> None:
    """
    Handles the PROCESS_CONTINUE entity in exclusive gateways by adding a "continue" entity to the gateway.
    The "continue" entity will point to the next task in the process (the first task after the exclusive gateway).
    Args:
        agent_task_pairs_to_add (list[dict]): A list of agent-task pairs to add to the gateways.
        gateways (list[dict]): A list of gateways.
        gateway (dict): The gateway to handle.
    Returns:
        None
    """
    assert gateway_contains_nested_gateways(gateway, gateways) == False
    for i in range(len(gateway["children"])):
        children_list = gateway["children"][i]
        if len(children_list) == 0:
            next_task_id = None
            for pair in agent_task_pairs_to_add:
                if pair["content"]["task"]["start"] > gateway["end"]:
                    next_task_id = pair["content"]["task"]["task_id"]
                    break
            gateway["children"][i].append(
                {"content": {"go_to": next_task_id}, "type": "continue"}
            )


def calculate_distance(gateway):
    return gateway["end"] - gateway["start"]


def nest_gateways(all_gateways: list[dict]) -> list[dict]:
    """
    Nests gateways according to their start and end indices.
    Args:
        all_gateways (list[dict]): A list of gateways.
    Returns:
        list[dict]: A list of nested gateways.
    """

    def is_nested(inner: dict, outer: dict) -> bool:
        """
        Checks if the inner gateway is nested within the outer gateway.
        If the inner gateway is not nested within the outer gateway, but the start and end indices of the inner gateway
        overlap significantly with the start and end indices of the outer gateway, the inner gateway is also considered
        nested within the outer gateway.
        Args:
            inner (dict): The inner gateway.
            outer (dict): The outer gateway.
        Returns:
            bool: True if the inner gateway is nested within the outer gateway, False otherwise.
        """
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

    def find_parent_and_path(gateway: dict) -> tuple[dict, dict]:
        """
        Finds the parent gateway and the path of the parent gateway that contains the given gateway.
        Args:
            gateway (dict): The gateway to find the parent gateway for.
        Returns:
            tuple[dict, dict]: The parent gateway and the path of the parent gateway that contains the given gateway.
        """
        parent = None
        parent_path = None
        for candidate in all_gateways:
            for path in candidate["paths"]:
                if candidate["id"] != gateway["id"] and is_nested(gateway, path):
                    if parent is None or is_nested(path, parent_path):
                        parent = candidate
                        parent_path = path
        return parent, parent_path

    def insert_in_sorted_order(children, gateway):
        index = 0
        if len(children) == 0:
            children.append(gateway)
            return
        child_start_idx = (
            children[0]["content"]["task"]["start"]
            if children[0]["type"] == "task"
            else children[0]["start"]
        )
        while index < len(children) and child_start_idx < gateway["start"]:
            child_start_idx = (
                children[index]["content"]["task"]["start"]
                if children[index]["type"] == "task"
                else children[index]["start"]
            )
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


def ranges_overlap_percentage(
    range1: tuple[int, int], range2: tuple[int, int], min_overlap_percentage=0.97
) -> bool:
    """
    Determines if two ranges overlap by a certain percentage. Each range is a tuple of the form (start, end).
    Args:
        range1 (tuple): The first range.
        range2 (tuple): The second range.
        min_overlap_percentage (float): The minimum percentage of overlap required for the ranges to be considered overlapping.
    Returns:
        bool: True if the ranges overlap by the minimum percentage, False otherwise.
    """

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
    process_info = []

    with open("output_logs/agent_task_pairs_final.json", "r") as file:
        agent_task_pairs = file.read()
    agent_task_pairs = json.loads(agent_task_pairs)

    with (open("output_logs/process_info_entities.json", "r")) as file:
        process_info = file.read()
    process_info = json.loads(process_info)

    if exists("output_logs/parallel_gateway_data.json"):
        with open("output_logs/parallel_gateway_data.json", "r") as file:
            parallel_gateway_data = file.read()
        parallel_gateway_data = json.loads(parallel_gateway_data)

    if exists("output_logs/exclusive_gateway_data.json"):
        with open("output_logs/exclusive_gateway_data.json", "r") as file:
            exclusive_gateway_data = file.read()
        exclusive_gateway_data = json.loads(exclusive_gateway_data)

    structure = create_bpmn_structure(
        agent_task_pairs, parallel_gateway_data, exclusive_gateway_data, process_info
    )
    write_to_file("bpmn_structure/bpmn_structure.json", structure)
