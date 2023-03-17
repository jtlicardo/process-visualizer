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
    return completion.choices[0].message


number_of_parallel_paths_template = """
Process description: '{}'

Based on this process, determine the number of parallel paths in the process. Respond with a single number in integer format. Do not respond with anything else.

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
        max_tokens=1,
    )

    print("Number of parallel paths:", completion.choices[0].message, "\n")
    return completion.choices[0].message


mark_up_parallel_paths_template = """
You will be given a description of a process with {} parallel paths. Your task is to mark up the text of the description so that the parallel paths can be easily identified, without modifying the content of the description. To achieve this, you should rewrite the process description by adding [S] at the start of each parallel path and [E] at the end of each parallel path. The [S] and [E] markers should be added inline with the rest of the text.

Here's an example of how the process description should be marked up:

Original process description: 'The process consists of three parallel paths. Path 1 involves A and B. Path 2 involves C and D. Path 3 involves E and F.'

Marked-up process description: 'The process consists of three parallel paths. [S] Path 1 involves A and B. [E] [S] Path 2 involves C and D. [E] [S] Path 3 involves E and F. [E]'

Process description: '{}'

Marked-up process description:
"""


def mark_up_parallel_paths(
    process_description: str, number_of_parallel_paths: int
) -> str:
    """
    Marks up the given process description so that the parallel paths can be easily identified.
    Args:
        process_description (str): A description of the business process
        number_of_parallel_paths (int): The number of parallel paths in the process
    Returns:
        str: The response from the GPT-3.5 model (the marked-up process description)
    """

    user_msg = mark_up_parallel_paths_template.format(
        number_of_parallel_paths, process_description
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print("Parallel paths:", completion.choices[0].message, "\n")
    return completion.choices[0].message
