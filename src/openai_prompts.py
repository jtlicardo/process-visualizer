import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

SYSTEM_MSG = "You are a highly experienced business process modelling expert, specializing in BPMN modelling. You will be provided with a description of a complex business process and will need to answer questions regarding the process. Your answers should be clear, accurate and concise."


def same_exclusive_gateway(process_description: str, condition_pair: str) -> str:
    """
    Determines whether the given condition pair should be modeled using the same exclusive gateway.
    Args:
        process_description (str): A description of the business process
        condition_pair (str): A pair of conditions to be evaluated, in the following format: 'condition1' and 'condition2'
    Returns:
        str: The response from the GPT-3.5 model (either 'TRUE' or 'FALSE')
    """

    same_exclusive_gateway_template = "Process description: '{}'\n\nBased on this process, determine whether the following conditions belong to the same exclusive gateway. Respond with either TRUE or FALSE\n\nCondition pair: {}\n\nResponse:"

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

    print(completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def extract_parallel_gateways(process_description: str) -> str:
    """
    Extracts the text which belongs to a specific parallel gateway.
    Args:
        process_description (str): A description of the business process
    Returns:
        str: The response from the GPT-3.5 model (the text which belongs to a specific parallel gateway)
    """

    extract_parallel_gateways_template = "You will receive a description of a process which contains one or more parallel gateways. Extract the text which belongs to a specific parallel gateway.\n\n###\n\nProcess: 'The professor sends the mail to the student. In the meantime, the student prepares his documents.'\nParallel gateway 1: The professor sends the mail to the student. In the meantime, the student prepares his documents.\n\nProcess: 'The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same. After the application has been approved, one team verifies the applicant's employment while another team verifies the applicant's income detail simultaneously. If both teams verify the information as accurate, the loan is approved and the process moves forward to the next step.'\nParallel gateway 1: The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same.\nParallel gateway 2: one team verifies the applicant's employment while another team verifies the applicant's income detail simultaneously.\n\nProcess: 'The process starts with the client discussing his ideas for the website. In the meantime, the agency presents potential solutions. After that, the developers start working on the project while the client meets with the representatives on a regular basis.'\nParallel gateway 1: the client discussing his ideas for the website. In the meantime, the agency presents potential solutions.\nParallel gateway 2: the developers start working on the project while the client meets with the representatives on a regular basis\n\nProcess: 'The manager sends the mail to the supplier and prepares the documents. At the same time, the customer searches for the goods and picks up the goods.'\nParallel gateway 1: The manager sends the mail to the supplier and prepares the documents. At the same time, the customer searches for the goods and picks up the goods.\n\nProcess: 'The process starts when a group of chefs generate ideas for new dishes. At this point, 3 things occur in parallel: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. The third path sees the accountants reviewing the potential cost of the dishes. Once each track has completed its analysis, the management reviews the findings of the analysis.'\nParallel gateway 1: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. The third path sees the accountants reviewing the potential cost of the dishes\n\nProcess: 'The employee delivers the package. In the meantime, the customer pays for the service. Finally, the customer opens the package while the employee delivers the next package.'\nParallel gateway 1: The employee delivers the package. In the meantime, the customer pays for the service.\nParallel gateway 2: the customer opens the package while the employee delivers the next package.\n\nProcess: 'The process starts with the student choosing his preference. In the meantime, the professor prepares the necessary paperwork. After that, the student starts his internship while the employer monitors the student's progress. Finally, the student completes his internship while the professor updates the database at the same time.'\nParallel gateway 1: The student choosing his preference. In the meantime, the professor prepares the necessary paperwork.\nParallel gateway 2: the student starts his internship while the employer monitors the student's progress.\nParallel gateway 3: the student completes his internship while the professor updates the database at the same time.\n\nProcess: 'The process starts when the employee starts the onboarding. In the meantime, the HR department handles the necessary paperwork. After that, the manager provides the employee with his initial tasks and monitors the employee's progress at the same time.'\nParallel gateway 1: the employee starts the onboarding. In the meantime, the HR department handles the necessary paperwork.\nParallel gateway 2: the manager provides the employee with his initial tasks and monitors the employee's progress at the same time.\n\nProcess: '{}'"

    user_msg = extract_parallel_gateways_template.format(process_description)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print(completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def number_of_parallel_paths(parallel_gateway: str) -> str:
    """
    Determines the number of parallel paths in the given parallel gateway.
    Args:
        parallel_gateway (str): A description of a parallel gateway
    Returns:
        str: The response from the GPT-3.5 model (a single integer)
    """

    number_of_parallel_paths_template = "You will receive a description of a parallel gateway. Determine the number of parallel paths in the parallel gateway. Respond with a single number in integer format.\n\n###\n\nParallel gateway: 'The R&D team researches and develops new technologies for the product. The next thing happening in parallel is the UX team designing the user interface and user experience. The interface has to be intuitive and user-friendly. The final thing occuring at the same time is when the QA team tests the product.'\nNumber of paths: 3\n\nParallel gateway: 'The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same.'\nNumber of paths: 2\n\nParallel gateway: '{}'\nNumber of paths:"

    user_msg = number_of_parallel_paths_template.format(parallel_gateway)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2,
    )

    print("Number of parallel paths:", completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def extract_3_parallel_paths(parallel_gateway: str) -> str:
    """
    Extracts the text of the parallel paths in the given process.
    Args:
        parallel_gateway (str): The text of the parallel gateway
    Returns:
        str: The response from the GPT-3.5 model (the text of the parallel paths)
    """

    extract_3_parallel_paths_template = "You will receive a process which contains 3 parallel paths.\nExtract the 3 spans of text that belong to the 3 parallel paths in the following format: <path> && <path> && <path>\nYou must extract the entire span of text that belongs to a given path, not just a part of it.\nUse the && symbols only twice.\n\n###\n\nProcess: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. The third path sees the accountants reviewing the potential cost of the dishes.\nPaths: the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe && the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes && the accountants reviewing the potential cost of the dishes\n\nProcess: The R&D team researches and develops new technologies for the product. The next thing happening in parallel is the UX team designing the user interface and user experience. The interface has to be intuitive and user-friendly. The final thing occuring at the same time is when the QA team tests the product.\nPaths: The R&D team researches and develops new technologies for the product && the UX team designing the user interface and user experience && the QA team tests the product\n\nProcess: {}\nPaths:"

    user_msg = extract_3_parallel_paths_template.format(parallel_gateway)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print("Parallel paths:", completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def extract_2_parallel_paths(parallel_gateway: str) -> str:
    """
    Extracts the text of the parallel paths in the given process.
    Args:
        parallel_gateway (str): The text of the parallel gateway
    Returns:
        str: The response from the GPT-3.5 model
    """

    extract_2_parallel_paths_template = "You will receive a process which contains 2 parallel paths.\nExtract the 2 spans of text that belong to the 2 parallel paths in the following format: <path> && <path>\nYou must extract the entire span of text that belongs to a given path, not just a part of it.\nUse the && symbols only once.\n\n###\n\nProcess: After that, he delivers the mail and greets people. Simultaneously, the milkman delivers milk.\nPaths: he delivers the mail and greets people && the milkman delivers milk\n\nProcess: There are 2 main things happening in parallel: the first thing is when John goes to the supermarket. The second thing is when Amy goes to the doctor. Amy also calls John at the same time. After those 2 main things are done, John goes home.\nPaths: John goes to the supermarket && Amy goes to the doctor. Amy also calls John at the same time.\n\nProcess: The team designs the interface. If it's approved, the team implements it. If not, the team revises the existing design and starts drafting a new one in parallel.\nPaths: the team revises the existing design && starts drafting a new one\n\nProcess: The process is composed of 2 activities done concurrently: the first one is the customer filling out a loan application. The second activity is a longer one, and it is composed of the manager deciding whether to prepare additional questions. If yes, the manager prepares additional questions. If the decision is not to prepare, the manager sends an email and manager reads the newspaper at the same time. After both activities have finished, the customer sends the application.\nPaths: the customer filling out a loan application && the manager deciding whether to prepare additional questions. If yes, the manager prepares additional questions. If the decision is not to prepare, the manager sends an email and manager reads the newspaper at the same time\n\nProcess: If the decision is not to prepare, the manager waits for the customer. After that, the manager sends an email. While sending an email, the manager also reads a newspaper.\nPaths: the manager sends an email && the manager also reads a newspaper\n\nProcess: {}\nPaths:"

    user_msg = extract_2_parallel_paths_template.format(parallel_gateway)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print("Parallel paths:", completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]
