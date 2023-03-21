import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

SYSTEM_MSG = "You are a highly experienced business process modelling expert, specializing in BPMN modelling. You have been provided with a \
description of a complex business process and are tasked with creating a BPMN diagram for it. You will be asked a series of specific \
questions related to the process and will need to make decisions on how to model each specific part of the BPMN diagram based on the \
provided business process description. Your answers should be clear and concise."

same_exclusive_gateway_template = """
Process description: '{}'

Based on this process, determine whether the following conditions belong to the same exclusive gateway. Respond with either TRUE or FALSE

Condition pair: {}

Response:
"""


def same_exclusive_gateway(process_description: str, condition_pair: str) -> str:
    """
    Determines whether the given condition pair should be modeled using the same exclusive gateway.
    Args:
        process_description (str): A description of the business process
        condition_pair (str): A pair of conditions to be evaluated, in the following format: 'condition1' and 'condition2'
    Returns:
        str: The response from the GPT-3.5 model (either 'TRUE' or 'FALSE')
    """

    user_msg = same_exclusive_gateway_template.format(
        process_description, condition_pair
    )

    print(condition_pair)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2,
    )

    print(completion.choices[0].message, "\n")
    return completion.choices[0].message["content"]


number_of_parallel_paths_template = """
Process description: '{}'

This is a process which contains activities executing in parallel. Determine the number of parallel paths in the process. Respond with a single number in integer format. Do not respond with anything else.

Example response: 2

Response:
"""


def number_of_parallel_paths(process_description: str) -> str:
    """
    Determines the number of parallel paths in the given process.
    Args:
        process_description (str): A description of the business process
    Returns:
        str: The response from the GPT-3.5 model (a single integer)
    """

    user_msg = number_of_parallel_paths_template.format(process_description)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2,
    )

    print("Number of parallel paths:", completion.choices[0].message, "\n")
    return completion.choices[0].message["content"]


extract_3_parallel_paths_template = """
You will be given a description of a process with 3 parallel paths. Extract the text of the parallel paths.

###

Process: The process begins by X doing Y. After that, the process splits into 3 parallel paths. Path 1 involves A and B. Path 2 involves C and D. Path 3 involves E and F. Once all these 3 activities are finished, A does B and the process ends.
Paths: Path 1 involves A and B || Path 2 involves C and D || Path 3 involves E and F

Process: {}
Paths:
"""


def extract_3_parallel_paths(process_description: str) -> str:
    """
    Extracts the text of the parallel paths in the given process.
    Args:
        process_description (str): A description of the business process
    Returns:
        str: The response from the GPT-3.5 model
    """

    prompt = extract_3_parallel_paths_template.format(process_description)

    completion = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=128, temperature=0
    )

    print("Parallel paths:", completion.choices[0], "\n")
    return completion.choices[0]["text"]


extract_2_parallel_paths_template = """
This is a process which contains 2 parallel paths. If the text contains nested parallel paths, extract the outermost parallel paths. Extract the text of 2 parallel paths in the following format: <path> || <path>

###

Process: John does A and John does B at the same time.
Paths: John does A || John does B
Process: After that, he delivers the mail and greets people. At the same time, the milkman delivers milk.
Paths: he delivers the mail and greets people || the milkman delivers milk
Process: There are 2 main things happening in parallel: the first thing is when A does B. The second thing is when C does D. C also does E in parallel. After those 2 main things are done, A does F.
Paths: A does B || C does D. C also does E in parallel.
Process: {}
Paths:
"""


def extract_2_parallel_paths(process_description: str) -> str:
    """
    Extracts the text of the parallel paths in the given process.
    Args:
        process_description (str): A description of the business process
    Returns:
        str: The response from the GPT-3.5 model
    """

    prompt = extract_2_parallel_paths_template.format(process_description)

    completion = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=128, temperature=0
    )

    print("Parallel paths:", completion.choices[0], "\n")
    return completion.choices[0]["text"]
