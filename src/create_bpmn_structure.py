from logging_utils import write_to_file


def create_bpmn_structure(
    agent_task_pairs, parallel_gateway_data, exclusive_gateway_data
):

    for pair in agent_task_pairs:
        pair["content"] = pair.copy()
        pair["type"] = "task" if "task" in pair else "loop"
        for key in pair.copy():
            if key != "type" and key != "content":
                del pair[key]

    agent_task_pairs_to_add = agent_task_pairs.copy()
    parallel_gateways = []
    exclusive_gateways = []
    structure = []

    if parallel_gateway_data is not None:
        parallel_gateways = create_parallel_gateways(
            parallel_gateway_data, agent_task_pairs, agent_task_pairs_to_add
        )

    if exclusive_gateway_data is not None:
        exclusive_gateways = create_exclusive_gateways(
            exclusive_gateway_data, agent_task_pairs, agent_task_pairs_to_add
        )

    for gateway in parallel_gateways:
        if "exclusive_parent" in gateway:
            add_nested_parallel_gateway(exclusive_gateways, parallel_gateways, gateway)

    for gateway in exclusive_gateways:
        if "parallel_parent" in gateway:
            add_nested_exclusive_gateway(parallel_gateways, exclusive_gateways, gateway)

    write_to_file("bpmn_structure/parallel_gateways.json", parallel_gateways)
    write_to_file("bpmn_structure/exclusive_gateways.json", exclusive_gateways)

    parallel_index = 0
    exclusive_index = 0

    for pair in agent_task_pairs_to_add:

        while (
            parallel_index < len(parallel_gateways)
            and parallel_gateways[parallel_index]["start"]
            < pair["content"]["task"]["start"]
        ):
            structure.append(parallel_gateways[parallel_index])
            parallel_index += 1

        while (
            exclusive_index < len(exclusive_gateways)
            and exclusive_gateways[exclusive_index]["start"]
            < pair["content"]["task"]["start"]
        ):
            structure.append(exclusive_gateways[exclusive_index])
            exclusive_index += 1

        structure.append(pair)

    # Add any remaining gateways to the structure
    structure.extend(parallel_gateways[parallel_index:])
    structure.extend(exclusive_gateways[exclusive_index:])

    return structure


def create_parallel_gateways(
    parallel_gateway_data, agent_task_pairs, agent_task_pairs_to_add
):

    parallel_gateways = []

    for gateway in parallel_gateway_data:
        children = []
        for i in range(len(gateway["paths"])):
            children.append([])
        for pair in agent_task_pairs:
            if (
                "parallel_gateway" in pair["content"]
                and pair["content"]["parallel_gateway"] == gateway["id"]
                and pair in agent_task_pairs_to_add
            ):
                children[pair["content"]["parallel_path"]].append(pair)
                agent_task_pairs_to_add.remove(pair)

        parallel_gateway = {
            "type": "parallel",
            "id": gateway["id"],
            "children": children,
            "start": gateway["start"],
            "end": gateway["end"],
        }

        if "exclusive_parent" in gateway:
            parallel_gateway["exclusive_parent"] = gateway["exclusive_parent"]
            parallel_gateway["exclusive_parent_path"] = gateway["exclusive_parent_path"]

        parallel_gateways.append(parallel_gateway)

    # Some lists in children may be empty due to nested parallel gateways
    # Replace empty lists with the nested parallel gateway
    for gateway in parallel_gateways:
        for i in range(len(gateway["children"])):
            if len(gateway["children"][i]) == 0:
                for pair in agent_task_pairs:
                    if (
                        "parent_gateway" in pair["content"]
                        and pair["content"]["parent_gateway"] == gateway["id"]
                        and pair["content"]["parent_path"] == i
                    ):
                        pg_id = pair["content"]["parallel_gateway"]
                        for pg in parallel_gateways:
                            if pg["id"] == pg_id:
                                gateway["children"][i] = [pg]
                                parallel_gateways.remove(pg)
                                break

    return parallel_gateways


def create_exclusive_gateways(
    exclusive_gateway_data, agent_task_pairs, agent_task_pairs_to_add
):

    exclusive_gateways = []

    for gateway in exclusive_gateway_data:
        children = []
        for i in range(len(gateway["paths"])):
            children.append([])
        for pair in agent_task_pairs:
            if (
                "exclusive_gateway_id" in pair["content"]
                and pair["content"]["exclusive_gateway_id"] == gateway["id"]
                and pair in agent_task_pairs_to_add
            ):
                children[pair["content"]["exclusive_gateway_path_id"]].append(pair)
                agent_task_pairs_to_add.remove(pair)

        exclusive_gateway = {
            "type": "exclusive",
            "id": gateway["id"],
            "children": children,
            "start": gateway["start"],
            "end": gateway["end"],
        }

        if "parallel_parent" in gateway:
            exclusive_gateway["parallel_parent"] = gateway["parallel_parent"]
            exclusive_gateway["parallel_parent_path"] = gateway["parallel_parent_path"]

        # Only the first agent pair in the children list can contain the "condition" key
        for i in range(len(exclusive_gateway["children"])):
            for j in range(len(exclusive_gateway["children"][i])):
                if (
                    j > 0
                    and "condition" in exclusive_gateway["children"][i][j]["content"]
                ):
                    del exclusive_gateway["children"][i][j]["content"]["condition"]

        if "parent_gateway_id" not in gateway:
            exclusive_gateways.append(exclusive_gateway)
        else:
            # Append the nested gateway to the specified path
            # If there are any tasks already in the path, append the nested gateway after the first task that comes before the nested gateway
            parent_gateway = next(
                (
                    pg
                    for pg in exclusive_gateways
                    if pg["id"] == gateway["parent_gateway_id"]
                ),
                None,
            )
            if parent_gateway is not None:
                for i in range(len(parent_gateway["children"])):
                    if (
                        len(parent_gateway["children"][i]) > 0
                        and i == gateway["parent_gateway_path_id"]
                    ):
                        for j in range(len(parent_gateway["children"][i])):
                            if (
                                parent_gateway["children"][i][j]["type"] == "task"
                                and parent_gateway["children"][i][j]["content"]["task"][
                                    "start"
                                ]
                                < gateway["start"]
                            ):
                                parent_gateway["children"][i].insert(
                                    j + 1, exclusive_gateway
                                )
                                break

    return exclusive_gateways


def add_nested_exclusive_gateway(parallel_gateways, exclusive_gateways, gateway):

    parent_gateway = next(
        (pg for pg in parallel_gateways if pg["id"] == gateway["parallel_parent"]),
        None,
    )

    assert parent_gateway is not None, "Parent exclusive gateway not found"

    parent_gateway["children"][gateway["parallel_parent_path"]].append(gateway)
    exclusive_gateways.remove(gateway)


def add_nested_parallel_gateway(exclusive_gateways, parallel_gateways, gateway):

    parent_gateway = next(
        (pg for pg in exclusive_gateways if pg["id"] == gateway["exclusive_parent"]),
        None,
    )
    assert parent_gateway is not None, "Parent parallel gateway not found"

    for i in range(len(gateway["children"])):
        if (
            len(gateway["children"][i]) > 0
            and "condition" in gateway["children"][i][0]["content"]
        ):
            gateway["condition"] = gateway["children"][i][0]["content"]["condition"]
            del gateway["children"][i][0]["content"]["condition"]

    parent_gateway["children"][gateway["exclusive_parent_path"]].append(gateway)
    parallel_gateways.remove(gateway)


if __name__ == "__main__":
    # Used for testing
    import json
    from os.path import exists

    parallel_gateway_data = None
    exclusive_gateway_data = None

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
    write_to_file("bpmn_structure.json", structure)
