import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

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

    system_msg = "You are a highly experienced business process modelling expert, specializing in BPMN modelling. You have been provided with a \
    description of a complex business process and are tasked with creating a BPMN diagram for it. You will be asked a series of specific \
    questions related to the process and will need to make decisions on how to model each specific part of the BPMN diagram based on the \
    provided business process description. Your answers should be clear and concise."

    user_msg = same_exclusive_gateway_template.format(
        process_description, condition_pair
    )

    print(condition_pair)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2,
    )

    print(completion.choices[0].message, "\n")
    return completion.choices[0].message
