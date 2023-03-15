import graphviz


class GraphGenerator:
    def __init__(self, data, format=None, notebook=False):

        self.bpmn = graphviz.Digraph("bpmn_diagram", filename="bpmn.gv")
        
        if format == "jpeg":
            self.bpmn.format = "jpeg"

        self.notebook = notebook

        self.data = data

        self.create_start_end_events = True

        self.last_completed_type = ""
        self.last_completed_type_id = 0  # i.e. counter

        self.tracker = {}

        self.task_counter = 0
        self.exclusive_gateway_counter = 0
        self.parallel_gateway_counter = 0

        self.bpmn.attr(
            "node", shape="box", style="filled", color="black", fillcolor="bisque"
        )

    def log_data(self, first_node, second_node):

        for node in [first_node, second_node]:
            if node not in self.tracker:
                self.tracker[node] = {"after": [], "before": []}

        self.tracker[first_node]["after"].append(second_node)
        self.tracker[second_node]["before"].append(first_node)

    def connect(self, first_node, second_node, label_parameter=None):
        if label_parameter is not None:
            self.bpmn.edge(first_node, second_node, label=label_parameter)
        else:
            self.bpmn.edge(first_node, second_node)
        self.log_data(first_node, second_node)

    def create_start_and_end_events(self):

        start_event_counter = 0
        end_event_counter = 0

        key = None

        self.bpmn.attr(
            "node",
            shape="ellipse",
            style="filled",
            color="black",
            fillcolor="lightpink",
        )

        for k, v in self.tracker.copy().items():
            if v["before"] == []:
                key = k
                self.bpmn.node(name=f"START_{start_event_counter}", label="START")
                self.connect(f"START_{start_event_counter}", str(key))
                start_event_counter += 1
            elif v["before"][0].startswith("T") and k == "T0":
                key = k
                self.bpmn.node(name=f"START_{start_event_counter}", label="START")
                self.connect(f"START_{start_event_counter}", str(key))
                start_event_counter += 1
            if k.startswith("EG") and k.endswith("S") and len(v["after"]) == 1:
                key = k
                self.bpmn.node(name=f"END_{end_event_counter}", label="END")
                self.connect(
                    str(key), f"END_{end_event_counter}", label_parameter="Else"
                )
                end_event_counter += 1
                continue
            elif v["after"] == []:
                key = k
                self.bpmn.node(name=f"END_{end_event_counter}", label="END")
                self.connect(str(key), f"END_{end_event_counter}")
                end_event_counter += 1

    def contains_nested_lists(self, list_parameter):
        assert isinstance(list_parameter, list)
        for item in list_parameter:
            if isinstance(item, list):
                return True
        return False

    def dictionary_is_element_of_list(self, dict_parameter, list_parameter):
        assert isinstance(dict_parameter, dict)
        assert isinstance(list_parameter, list)
        for item in list_parameter:
            if isinstance(item, dict) and item == dict_parameter:
                return True
        return False

    def get_nested_lists(self, list_parameter):
        assert isinstance(list_parameter, list)
        nested_lists = []
        for item in list_parameter:
            if isinstance(item, list):
                nested_lists.append(item)
        return nested_lists

    def dict_is_first_element(self, dictionary, list_of_lists):
        assert isinstance(dictionary, dict)
        assert isinstance(list_of_lists, list)
        for lst in list_of_lists:
            if isinstance(lst[0], dict) and lst[0] == dictionary:
                return True
        return False

    def dict_is_direct_child(self, element, gateway):
        assert isinstance(element, dict)
        assert isinstance(gateway, dict)
        if not self.contains_nested_lists(gateway["children"]):
            return True
        else:
            if self.dictionary_is_element_of_list(element, gateway["children"]):
                return True
            else:
                nested_lists = self.get_nested_lists(gateway["children"])
                if self.dict_is_first_element(element, nested_lists):
                    return True
        return False

    def count_conditions_in_gateway(self, gateway):
        count = 0
        for child in gateway["children"]:
            if isinstance(child, list):
                if child[0]["type"] == "parallel" or child[0]["type"] == "exclusive":
                    if "condition" in child[0]:
                        count += 1
                elif child[0]["type"] == "task":
                    if "condition" in child[0]["content"]:
                        count += 1
            elif "content" in child and "condition" in child["content"]:
                count += 1
        return count

    def check_for_loops_in_gateway(self, gateway):
        assert isinstance(gateway, dict)
        for child in gateway["children"]:
            if isinstance(child, list):
                for element in child:
                    if element["type"] == "loop":
                        return True
            else:
                if child["type"] == "loop":
                    return True
        return False

    def check_for_loops_in_list(self, lst):
        assert isinstance(lst, list)
        for element in lst:
            if element["type"] == "loop":
                return True
        return False

    def get_last_task_in_gateway_in_path_with_no_loops(self, gateway):
        assert isinstance(gateway, dict)
        last_task = None
        for child in gateway["children"]:
            if isinstance(child, list):
                if not self.check_for_loops_in_list(child):
                    for element in child:
                        if element["type"] == "task":
                            last_task = element
            else:
                if child["type"] == "task":
                    last_task = child
        return last_task

    def create_node(self, element, type, agent=None, task=None):

        assert type == "T" or type == "E" or type == "P"

        if type == "T":
            assert agent is not None and task is not None
            self.bpmn.node(name=f"T{self.task_counter}", label=f"{agent}: {task}")
            element["id"] = f"T{self.task_counter}"
            self.task_counter += 1
            self.tracker[element["id"]] = {"after": [], "before": []}
        else:
            counter = 0
            if type == "E":
                counter = self.exclusive_gateway_counter
                label = "X"
                self.exclusive_gateway_counter += 1
            else:
                counter = self.parallel_gateway_counter
                label = "+"
                self.parallel_gateway_counter += 1

            self.bpmn.attr(
                "node",
                shape="diamond",
                style="filled",
                color="black",
                fillcolor="azure",
            )

            self.bpmn.node(name=f"{type}G{counter}_S", label=label)

            if "single_condition" not in element and "has_loops" not in element:
                self.bpmn.node(name=f"{type}G{counter}_E", label=label)

            element["id"] = f"{type}G{counter}"

            self.bpmn.attr(
                "node", shape="box", style="filled", color="black", fillcolor="bisque"
            )

    def handle_task(
        self,
        element,
        local_index=None,
        last=False,
        parent_gateway=None,
        previous_element=None,
    ):

        agent = element["content"]["agent"]["word"]
        task = element["content"]["task"]["word"]

        self.create_node(type="T", element=element, agent=agent, task=task)

        if "condition" in element["content"]:
            self.connect(
                f"EG{self.exclusive_gateway_counter - 1}_S",
                f"T{self.task_counter - 1}",
                label_parameter=element["content"]["condition"]["word"],
            )

        if parent_gateway is not None:
            if parent_gateway["type"] == "parallel" and self.dict_is_direct_child(
                element, parent_gateway
            ):
                self.connect(
                    f"{parent_gateway['id']}_S",
                    f"T{self.task_counter - 1}",
                )

        if (
            isinstance(local_index, int)
            and local_index != 0
            and "condition" not in element["content"]
            and not self.dict_is_direct_child(element, parent_gateway)
        ):
            self.connect(f"T{self.task_counter - 2}", f"T{self.task_counter - 1}")
        elif local_index is None and self.last_completed_type == "task":
            self.connect(f"T{self.task_counter - 2}", f"T{self.task_counter - 1}")

        if previous_element is not None and not isinstance(previous_element, list):
            if (
                previous_element["type"] == "exclusive"
                or previous_element["type"] == "parallel"
            ):
                if "has_loops" not in previous_element:
                    if "single_condition" not in previous_element:
                        self.connect(
                            f"{previous_element['id']}_E",
                            f"{element['id']}",
                        )
                    else:
                        last_child = previous_element["children"][0][-1]
                        if last_child["type"] != "task":
                            self.connect(
                                f"{last_child['id']}_E",
                                f"{element['id']}",
                            )
                        else:
                            self.connect(
                                f"{last_child['id']}",
                                f"{element['id']}",
                            )
                else:
                    last_task = self.get_last_task_in_gateway_in_path_with_no_loops(
                        previous_element
                    )
                    self.connect(
                        f"{last_task['id']}",
                        f"{element['id']}",
                    )

        if last:
            if (
                parent_gateway is not None
                and "single_condition" not in parent_gateway
                and "has_loops" not in parent_gateway
            ):
                self.connect(
                    f"{element['id']}",
                    f"{parent_gateway['id']}_E",
                )

        self.last_completed_type = "task"
        self.last_completed_type_id = self.task_counter - 1

    def handle_list(self, data, parent_gateway):

        for index, element in enumerate(data):

            last = True if index == len(data) - 1 else False

            if index != 0:
                previous_element = data[index - 1]
            else:
                previous_element = None

            if element["type"] == "task":
                self.handle_task(
                    element=element,
                    local_index=index,
                    last=last,
                    parent_gateway=parent_gateway,
                    previous_element=previous_element,
                )
            elif element["type"] == "loop":
                assert parent_gateway["type"] == "exclusive"
                assert previous_element is not None
                assert last is True
                self.connect(f"{previous_element['id']}", element["content"]["go_to"])
            elif element["type"] == "exclusive":
                self.handle_gateway(
                    element=element,
                    type="exclusive",
                    last=last,
                    parent_gateway=parent_gateway,
                    previous_element=previous_element,
                )
            else:
                self.handle_gateway(
                    element=element,
                    type="parallel",
                    last=last,
                    parent_gateway=parent_gateway,
                    previous_element=previous_element,
                )

    def handle_gateway(
        self, element, type, last=None, parent_gateway=None, previous_element=None
    ):

        assert type == "parallel" or type == "exclusive"

        if type == "exclusive":
            num = self.count_conditions_in_gateway(element)
            if num == 1:
                element["children"] = [element["children"]]
                element["single_condition"] = True

        if self.check_for_loops_in_gateway(element):
            element["has_loops"] = True

        self.create_node(
            type="P", element=element
        ) if type == "parallel" else self.create_node(type="E", element=element)

        if "condition" in element:
            assert parent_gateway is not None
            assert parent_gateway["type"] == "exclusive"
            condition = element["condition"]["word"]
            self.connect(
                f"{parent_gateway['id']}_S",
                f"{element['id']}_S",
                label_parameter=condition,
            )

        for index, child in enumerate(element["children"]):

            if index != 0:
                previous_child = element["children"][index - 1]
            else:
                previous_child = None

            if isinstance(child, list):
                self.handle_list(data=child, parent_gateway=element)
            else:
                if child["type"] == "task":
                    self.handle_task(
                        element=child,
                        local_index=index,
                        last=True,
                        parent_gateway=element,
                        previous_element=previous_child,
                    )
                elif child["type"] == "loop":
                    assert "condition" in child["content"]
                    self.connect(
                        f"{element['id']}_S",
                        child["content"]["go_to"],
                        label_parameter=child["content"]["condition"]["word"],
                    )
                elif child["type"] == "parallel":
                    self.handle_gateway(
                        element=element,
                        type="parallel",
                        last=True,
                        parent_gateway=element,
                        previous_element=previous_child,
                    )
                else:
                    self.handle_gateway(
                        element=element,
                        type="exclusive",
                        last=True,
                        parent_gateway=element,
                        previous_element=previous_child,
                    )

        if parent_gateway is not None:
            assert "id" in parent_gateway
            if last and "single_condition" not in parent_gateway:
                self.connect(
                    f"{element['id']}_E",
                    f"{parent_gateway['id']}_E",
                )

        if previous_element is not None:
            if previous_element["type"] == "task":
                self.connect(
                    f"{previous_element['id']}",
                    f"{element['id']}_S",
                )

        self.last_completed_type = type
        self.last_completed_type_id = (
            self.exclusive_gateway_counter - 1
            if type == "exclusive"
            else self.parallel_gateway_counter - 1
        )

    def generate_graph(self):

        if isinstance(self.data, dict):
            self.data = [self.data]

        for global_index, element in enumerate(self.data):

            if global_index != 0:
                previous_element = self.data[global_index - 1]
            else:
                previous_element = None

            if element["type"] == "task":
                self.handle_task(element=element, previous_element=previous_element)
            elif element["type"] == "parallel":
                self.handle_gateway(
                    element=element, type="parallel", previous_element=previous_element
                )
            else:
                self.handle_gateway(
                    element=element, type="exclusive", previous_element=previous_element
                )

            if global_index == len(self.data) - 1 and self.create_start_end_events:
                self.create_start_and_end_events()

    def show(self):
        if self.notebook == True:
            self.bpmn.save()
        else:
            self.bpmn.view()

    def save_file(self):
        self.bpmn.render(outfile="./src/bpmn.jpeg")

if __name__ == "__main__":
    # Used for debugging purposes
    import json
    with open("output_logs/final_output.txt", 'r') as file:
        content = file.read()
    data = json.loads(content)
    bpmn = GraphGenerator(data, notebook=False)
    bpmn.generate_graph()
    bpmn.show()
