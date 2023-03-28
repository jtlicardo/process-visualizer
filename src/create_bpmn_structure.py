from logging_utils import write_to_file


def create_bpmn_structure(
    agent_task_pairs, parallel_gateway_data, exclusive_gateway_data
):

    for pair in agent_task_pairs:
        pair["content"] = pair.copy()
        pair["type"] = "task"
        for key in pair.copy():
            if key != "type" and key != "content":
                del pair[key]

    parallel_gateways = []
    exclusive_gateways = []

    structure = []

    if parallel_gateway_data is not None:
        for gateway in parallel_gateway_data:
            children = []
            for i in range(len(gateway["paths"])):
                children.append([])
            for pair in agent_task_pairs:
                if (
                    "parallel_gateway" in pair["content"]
                    and pair["content"]["parallel_gateway"] == gateway["id"]
                ):
                    children[pair["content"]["parallel_path"]].append(pair)
            parallel_gateway = {
                "type": "parallel",
                "id": gateway["id"],
                "children": children,
            }
            parallel_gateways.append(parallel_gateway)

        # Some lists in children may be empty due to nested parallel gateways
        # Replace empty lists with the id of the nested parallel gateway
        for gateway in parallel_gateways:
            for i in range(len(gateway["children"])):
                if len(gateway["children"][i]) == 0:
                    for pair in agent_task_pairs:
                        if (
                            "parent_gateway" in pair["content"]
                            and pair["content"]["parent_gateway"] == gateway["id"]
                            and pair["content"]["parent_path"] == i
                        ):
                            gateway["children"][i] = pair["content"]["parallel_gateway"]

        # Replace the ids of the nested parallel gateways with the actual gateways; remove the gateway from the list
        for gateway in parallel_gateways:
            for i in range(len(gateway["children"])):
                if isinstance(gateway["children"][i], str):
                    for pg in parallel_gateways:
                        if pg["id"] == gateway["children"][i]:
                            gateway["children"][i] = [pg]
                            parallel_gateways.remove(pg)

        write_to_file("bpmn_structure/parallel_gateways.json", parallel_gateways)

    if exclusive_gateway_data is not None:
        for gateway in exclusive_gateway_data:
            children = []
            for i in range(len(gateway["paths"])):
                children.append([])
            for pair in agent_task_pairs:
                if (
                    "exclusive_gateway_id" in pair["content"]
                    and pair["content"]["exclusive_gateway_id"] == gateway["id"]
                ):
                    children[pair["content"]["exclusive_gateway_path_id"]].append(pair)
            exclusive_gateway = {
                "type": "exclusive",
                "id": gateway["id"],
                "children": children,
            }

            # Only the first agent pair in the children list can contain the "condition" key
            for i in range(len(exclusive_gateway["children"])):
                for j in range(len(exclusive_gateway["children"][i])):
                    if (
                        j > 0
                        and "condition"
                        in exclusive_gateway["children"][i][j]["content"]
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
                                    and parent_gateway["children"][i][j]["content"][
                                        "task"
                                    ]["start"]
                                    < gateway["start"]
                                ):
                                    parent_gateway["children"][i].insert(
                                        j + 1, exclusive_gateway
                                    )
                                    break

        write_to_file("bpmn_structure/exclusive_gateways.json", exclusive_gateways)

    added = []

    for pair in agent_task_pairs:
        if "parallel_gateway" in pair["content"]:
            if pair["content"]["parallel_gateway"] in added:
                continue
            elif "parent_gateway" in pair:
                if pair["content"]["parent_gateway"] in added:
                    continue
            else:
                # Append the parallel gateway with the corresponding id
                for gateway in parallel_gateways:
                    if gateway["id"] == pair["content"]["parallel_gateway"]:
                        structure.append(gateway)
                        added.append(gateway["id"])
        elif "exclusive_gateway_id" in pair["content"]:
            if pair["content"]["exclusive_gateway_id"] in added:
                continue
            else:
                # Append the exclusive gateway with the corresponding id
                for gateway in exclusive_gateways:
                    if gateway["id"] == pair["content"]["exclusive_gateway_id"]:
                        structure.append(gateway)
                        added.append(gateway["id"])
        else:
            structure.append(pair)

    return structure


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
