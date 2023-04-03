import configparser
import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

SYSTEM_MSG = "You are a highly experienced business process modelling expert, specializing in BPMN modelling. You will be provided with a description of a complex business process and will need to answer questions regarding the process. Your answers should be clear, accurate and concise."


def get_model():
    config = configparser.ConfigParser()
    config.read("src\config.ini")
    return config["OPENAI"]["model"]


def extract_exclusive_gateways(process_description: str) -> str:

    extract_exclusive_gateways_template = "You will receive a description of a process which contains conditions. Extract the text which belongs to a specific exclusive gateway.\n\n###\n\nProcess: 'If the client opts for funding, they will have to complete a loan request. Then, the client submits the application to the financial institution. If the client decides to pay with currency, they will need to bring the full amount of the vehicle's cost to the dealership to finalize the purchase. Once the client has chosen to fund or pay with currency, they must sign the agreement before concluding the transaction.'\nExclusive gateway 1: If the client opts for funding, they will have to complete a loan request. Then, the client submits the application to the financial institution. If the client decides to pay with currency, they will need to bring the full amount of the vehicle's cost to the dealership to finalize the purchase.\n\nProcess: 'If the customer chooses to finance, the customer will need to fill out a loan application. If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to the dealership in order to complete the transaction. After the financial decision has been made, if the customer decides to trade in their old car, the dealership will provide an appraisal and deduct the value from the total cost of the new car. However, if the customer chooses not to trade in their old car, they will need to pay the full price of the new car.'\nExclusive gateway 1: If the customer chooses to finance, the customer will need to fill out a loan application. If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to the dealership in order to complete the transaction.\nExclusive gateway 2: if the customer decides to trade in their old car, the dealership will provide an appraisal and deduct the value from the total cost of the new car. However, if the customer chooses not to trade in their old car, they will need to pay the full price of the new car.\n\nProcess: 'If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.'\nExclusive gateway 1: If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.\n\nProcess: 'If the company chooses to create a new product, the company designs the product. If the company is satisfied with the design, the company launches the product and the process ends. If not, the company redesigns the product. On the other hand, if the company chooses to modify an existing product, the company chooses a product to redesign and then redesigns the product.'\nExclusive gateway 1: If the company chooses to create a new product, the company designs the product. If the company is satisfied with the design, the company launches the product and the process ends. If not, the company redesigns the product. On the other hand, if the company chooses to modify an existing product, the company chooses a product to redesign and then redesigns the product.\nExclusive gateway 2: If the company is satisfied with the design, the company launches the product and the process ends. If not, the company redesigns the product.\n\nProcess: '{}'"

    user_msg = extract_exclusive_gateways_template.format(process_description)

    model = get_model()

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print(f'{completion["usage"]["total_tokens"]} tokens used ({model})')

    print(completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def extract_gateway_conditions(process_description: str, conditions: str) -> str:

    extract_gateway_conditions_template = "You will receive a description of a process and a list of conditions that appear in the process. Determine which conditions belong to which exclusive gateway. Determine which conditions belong to which exclusive gateway. Use only the conditions that are listed, do not take anything else from the process description.\n\n###\n\nProcess: 'The customer decides if he wants to finance or pay in cash. If the customer chooses to finance, the customer will need to fill out a loan application. If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to the dealership in order to complete the transaction.'\nConditions: If the customer chooses to finance', 'If the customer chooses to pay in cash'\nExclusive gateway 1: If the customer chooses to finance || If the customer chooses to pay in cash\n\nProcess: 'The restaurant receives the food order from the customer. If the dish is not available, the customer is informed that the order cannot be fulfilled. If the dish is available and the payment is successful, the restaurant prepares and serves the order. If the dish is available, but the payment fails, the customer is notified that the order cannot be processed.'\nConditions: 'If the dish is not available', 'If the dish is available and the payment is successful', 'If the dish is available, but the payment fails'\nExclusive gateway 1: If the dish is not available || If the dish is available and the payment is successful || If the dish is available, but the payment fails\n\nProcess: 'The customer places an order on the website. The system checks the inventory status of the ordered item. If the item is in stock, the system checks the customer's payment information. If the item is out of stock, the system sends an out of stock notification to the customer and cancels the order. After checking the customer's payment info, if the payment is authorized, the system generates an order confirmation and sends it to the customer, and the order is sent to the warehouse for shipping. If the payment is declined, the system sends a payment declined notification to the customer and cancels the order.'\nConditions: 'If the item is in stock', 'If the item is out of stock', 'if the payment is authorized', 'If the payment is declined'\nExclusive gateway 1: If the item is in stock || If the item is out of stock\nExclusive gateway 2: if the payment is authorized || If the payment is declined\n\nProcess: 'The process begins with the student choosing his preferences. Then the professor allocates the student. After that the professor notifies the student. The employer evaluates the candidate. If the student is accepted, the professor notifies the student. The student then completes his internship. If the student is successful, he gets a passing grade'\nConditions: 'If the student is accepted','If the student is successful'\nExclusive gateway 1: If the student is accepted\nExclusive gateway 2: If the student is successful\n\nProcess: {}\nConditions: {}"

    user_msg = extract_gateway_conditions_template.format(
        process_description, conditions
    )

    print(conditions)

    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_msg,
        max_tokens=128,
        temperature=0,
    )

    print(f'{completion["usage"]["total_tokens"]} tokens used (text-davinci-003)')

    print(completion.choices[0]["text"], "\n")
    return completion.choices[0]["text"]


def extract_parallel_gateways(process_description: str) -> str:

    extract_parallel_gateways_template = "You will receive a description of a process which contains one or more parallel gateways. Extract the text which belongs to a specific parallel gateway.\n\n###\n\nProcess: 'The professor sends the mail to the student. In the meantime, the student prepares his documents.'\nParallel gateway 1: The professor sends the mail to the student. In the meantime, the student prepares his documents.\n\nProcess: 'The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same. After the application has been approved, one team verifies the applicant's employment while another team verifies the applicant's income detail simultaneously. If both teams verify the information as accurate, the loan is approved and the process moves forward to the next step.'\nParallel gateway 1: The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same.\nParallel gateway 2: one team verifies the applicant's employment while another team verifies the applicant's income detail simultaneously.\n\nProcess: 'The process starts with the client discussing his ideas for the website. In the meantime, the agency presents potential solutions. After that, the developers start working on the project while the client meets with the representatives on a regular basis.'\nParallel gateway 1: the client discussing his ideas for the website. In the meantime, the agency presents potential solutions.\nParallel gateway 2: the developers start working on the project while the client meets with the representatives on a regular basis\n\nProcess: 'The manager sends the mail to the supplier and prepares the documents. At the same time, the customer searches for the goods and picks up the goods.'\nParallel gateway 1: The manager sends the mail to the supplier and prepares the documents. At the same time, the customer searches for the goods and picks up the goods.\n\nProcess: 'The process starts when a group of chefs generate ideas for new dishes. At this point, 3 things occur in parallel: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. The third path sees the accountants reviewing the potential cost of the dishes. Once each track has completed its analysis, the management reviews the findings of the analysis.'\nParallel gateway 1: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. The third path sees the accountants reviewing the potential cost of the dishes\n\nProcess: 'The employee delivers the package. In the meantime, the customer pays for the service. Finally, the customer opens the package while the employee delivers the next package.'\nParallel gateway 1: The employee delivers the package. In the meantime, the customer pays for the service.\nParallel gateway 2: the customer opens the package while the employee delivers the next package.\n\nProcess: 'The project manager defines the requirements. The process then splits into 2 parallel paths: in the first path the front-end development team designs the user interface. If the design is approved, the team implements it. If not, the team revises the design and continues to implement the approven parts of the design at the same time. In the second parallel path the back-end development team builds the server-side functionality of the app. After the two parallel paths merge, the QA team test the app's performance.'\nParallel gateway 1: in the first path the front-end development team designs the user interface. If the design is approved, the team implements it. If not, the team revises the design and continues to implement the approven parts of the design at the same time. In the second parallel path the back-end development team builds the server-side functionality of the app.\n\nProcess: 'The process starts with the student choosing his preference. In the meantime, the professor prepares the necessary paperwork. After that, the student starts his internship while the employer monitors the student's progress. Finally, the student completes his internship while the professor updates the database at the same time.'\nParallel gateway 1: The student choosing his preference. In the meantime, the professor prepares the necessary paperwork.\nParallel gateway 2: the student starts his internship while the employer monitors the student's progress.\nParallel gateway 3: the student completes his internship while the professor updates the database at the same time.\n\nProcess: 'The process starts when the employee starts the onboarding. In the meantime, the HR department handles the necessary paperwork. After that, the manager provides the employee with his initial tasks and monitors the employee's progress at the same time.'\nParallel gateway 1: the employee starts the onboarding. In the meantime, the HR department handles the necessary paperwork.\nParallel gateway 2: the manager provides the employee with his initial tasks and monitors the employee's progress at the same time.\n\nProcess: '{}'"

    user_msg = extract_parallel_gateways_template.format(process_description)

    model = get_model()

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print(f'{completion["usage"]["total_tokens"]} tokens used ({model})')

    print(completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def number_of_parallel_paths(parallel_gateway: str) -> str:

    number_of_parallel_paths_template = "You will receive a description of a parallel gateway. Determine the number of parallel paths in the parallel gateway. Respond with a single number in integer format.\n\n###\n\nParallel gateway: 'The R&D team researches and develops new technologies for the product. The next thing happening in parallel is the UX team designing the user interface and user experience. The interface has to be intuitive and user-friendly. The final thing occuring at the same time is when the QA team tests the product.'\nNumber of paths: 3\n\nParallel gateway: 'The credit analyst evaluates the creditworthiness and collateral of the applicant. Meanwhile, another team does the same.'\nNumber of paths: 2\n\nParallel gateway: '{}'\nNumber of paths:"

    user_msg = number_of_parallel_paths_template.format(parallel_gateway)

    model = get_model()

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2,
    )

    print(f'{completion["usage"]["total_tokens"]} tokens used ({model})')

    print("Number of parallel paths:", completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def extract_parallel_tasks(sentence: str) -> str:

    extract_parallel_tasks_template = 'You will receive a sentence that contains multiple tasks being done in parallel.\nExtract the tasks being done in parallel in the following format (the number of tasks may vary):\nTask 1: <task>\nTask 2: <task>\n\n###\n\nSentence: "The chef is simultaneously preparing the entree and dessert dishes."\nTask 1: prepare the entree\nTask 2: prepare the dessert dishes\n\nSentence: "The chef chops the vegetables, stirs the soup, and adds spices to the pot simultaneously."\nTask 1: chop the vegetables\nTask 2: stir the soup\nTask 3: add spices\n\nSentence: "The project manager is coordinating with the design team and development team concurrently."\nTask 1: coordinate with the design team\nTask 2: coordinate with the development team\n\nSentence: "{}"'

    user_msg = extract_parallel_tasks_template.format(sentence)

    model = "gpt-3.5-turbo"

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    print(f'{completion["usage"]["total_tokens"]} tokens used ({model})')

    print("Parallel tasks:", completion.choices[0].message["content"], "\n")
    return completion.choices[0].message["content"]


def extract_3_parallel_paths(parallel_gateway: str) -> str:

    extract_3_parallel_paths_template = "You will receive a process which contains 3 parallel paths.\nExtract the 3 spans of text that belong to the 3 parallel paths in the following format: <path> && <path> && <path>\nYou must extract the entire span of text that belongs to a given path, not just a part of it.\nUse the && symbols only twice.\n\n###\n\nProcess: the first thing is the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe. The second path involves the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes. The third path sees the accountants reviewing the potential cost of the dishes.\nPaths: the kitchen team analyzing the ideas for practicality. The kitchen team also creates the recipe && the customer service team conducting market research for the dishes. At the same time, the art team creates visual concepts for the potential dishes && the accountants reviewing the potential cost of the dishes\n\nProcess: The R&D team researches and develops new technologies for the product. The next thing happening in parallel is the UX team designing the user interface and user experience. The interface has to be intuitive and user-friendly. The final thing occuring at the same time is when the QA team tests the product.\nPaths: The R&D team researches and develops new technologies for the product && the UX team designing the user interface and user experience && the QA team tests the product\n\nProcess: {}\nPaths:"

    user_msg = extract_3_parallel_paths_template.format(parallel_gateway)

    model = get_model()

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    paths = completion.choices[0].message["content"]
    print(f'{completion["usage"]["total_tokens"]} tokens used ({model})')

    assert (
        "&&" in paths and paths.count("&&") == 2
    ), "Model did not return 3 parallel paths"

    print("Parallel paths:", paths, "\n")
    return completion.choices[0].message["content"]


def extract_2_parallel_paths(parallel_gateway: str) -> str:

    extract_2_parallel_paths_template = "You will receive a process which contains 2 parallel paths.\nExtract the 2 spans of text that belong to the 2 parallel paths in the following format: <path> && <path>\nYou must extract the entire span of text that belongs to a given path, not just a part of it.\nUse the && symbols exactly once.\n\n###\n\nProcess: After that, he delivers the mail and greets people. Simultaneously, the milkman delivers milk.\nPaths: he delivers the mail and greets people && the milkman delivers milk\n\nProcess: There are 2 main things happening in parallel: the first thing is when John goes to the supermarket. The second thing is when Amy goes to the doctor. Amy also calls John at the same time. After those 2 main things are done, John goes home.\nPaths: John goes to the supermarket && Amy goes to the doctor. Amy also calls John at the same time.\n\nProcess: The team designs the interface. If it's approved, the team implements it. If not, the team revises the existing design and starts drafting a new one in parallel.\nPaths: the team revises the existing design && starts drafting a new one\n\nProcess: in the first path the front-end development team designs the user interface. If the design is approved, the front-end development team implements it. If not, the front-end development team revises it and continues to implement the approven parts of the design at the same time. In the second parallel path the front-end development team builds the server-side functionality of the mobile app.\nPaths: the front-end development team designs the user interface. If the design is approved, the front-end development team implements it. If not, the front-end development team revises it and continues to implement the approven parts of the design at the same time. && the front-end development team builds the server-side functionality of the mobile app.\n\nProcess: the team designs the user interface. If the design is approved, the team implements the design. If not, the team revises the design and continues to implement the approven parts of the design at the same time.\nPaths: the team revises the design && continues to implement the approven parts of the design\n\nProcess: The process is composed of 2 activities done concurrently: the first one is the customer filling out a loan application. The second activity is a longer one, and it is composed of the manager deciding whether to prepare additional questions. If yes, the manager prepares additional questions. If the decision is not to prepare, the manager sends an email and manager reads the newspaper at the same time. After both activities have finished, the customer sends the application.\nPaths: the customer filling out a loan application && the manager deciding whether to prepare additional questions. If yes, the manager prepares additional questions. If the decision is not to prepare, the manager sends an email and manager reads the newspaper at the same time\n\nProcess: If the decision is not to prepare, the manager waits for the customer. After that, the manager sends an email. While sending an email, the manager also reads a newspaper.\nPaths: the manager sends an email && the manager also reads a newspaper\n\nProcess: {}\nPaths:"

    user_msg = extract_2_parallel_paths_template.format(parallel_gateway)

    model = get_model()

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )

    paths = completion.choices[0].message["content"]
    print(f'{completion["usage"]["total_tokens"]} tokens used ({model})')

    assert "&&" in paths, "Model did not return 2 parallel paths"

    print("Parallel paths:", paths, "\n")
    return paths
