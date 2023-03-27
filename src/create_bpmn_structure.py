from logging_utils import write_to_file


def create_bpmn_structure(
    agent_task_pairs, parallel_gateway_data, exclusive_gateway_data
):

    for pair in agent_task_pairs:
        pair["type"] = "task"
        pair["content"] = pair.copy()
        for key in pair.copy():
            if key != "type" and key != "content":
                del pair[key]

    parallel_gateways = []
    exclusive_gateways = []

    structure = []

    # [
    # {"id": "PG0", "start": 0, "end": 100, "paths": [
    # {"start": 0, "end": 50},
    # {"start": 50, "end": 100}
    # ]}
    # ]
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

    # [{'id': 'EG0', 'conditions': ['If the company chooses to create a new product',
    # 'If the company chooses to modify an existing product'], 'start': 54, 'end': 359},
    # "condition_indices": [{'start': 54, 'end': 101}, {'start': 300, 'end': 359}]
    if exclusive_gateway_data is not None:
        pass

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
            continue
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
