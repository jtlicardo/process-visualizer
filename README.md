# Process visualizer

This is an application that aims to convert textual descriptions of processes into simplified BPMN diagrams using [Graphviz](https://graphviz.org/).

The app supports the following BPMN elements:

* tasks
* exclusive gateways
* parallel gateways
* start and end events

## How to run

1. Clone the repo
1. Install the required dependencies: `pip install -r requirements.txt`
1. Download the necessary spaCy models:  
    `python -m spacy download en_core_web_sm`  
    `python -m spacy download en_core_web_md`
1. Create an .env file in the root of the `src` folder with your OpenAI API key as an environment variable: `OPENAI_KEY=<your_key>`
1. Run the script by running `python .\src\main.py -t "Textual description of the process"`. Alternatively, you can run the script by providing a path to a file containing the textual description of the process: `python .\src\main.py -f <path_to_file.txt> `

## Example inputs and outputs

### Example #1

*The process begins when the student logs in to the university's website. He then takes an online exam. After that, the system grades it. If the student scores below 60%, he takes the exam again. If the student scores 60% or higher on the exam, the professor enters the grade.*

<details>
  <summary><b>Show output</b></summary>
  
<p align="center">
<img src="images/image_1.png" width="600">
</p>
</details>

---

### Example #2

*The customer decides if he wants to finance or pay in cash. If the customer chooses to finance, the customer will need to fill out a loan application. After that, the customer sends the application to the bank. If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to the dealership in order to complete the transaction. After the customer has chosen to finance or pay in cash, the customer must sign the contract before the transaction is completed.*

<details>
  <summary><b>Show output</b></summary>
  
<p align="center">
<img src="images/image_2.png" width="600">
</p>
</details>

---

### Example #3

*The manager sends the mail to the supplier and prepares the documents. At the same time, the
customer searches for the goods and picks up the goods.*

<details>
  <summary><b>Show output</b></summary>
  
<p align="center">
<img src="images/image_3.png" width="600">
</p>
</details>

---

### Example #4

*The customer decides if he wants to finance or pay in cash. If the customer chooses to finance, two activities will happen in parallel: the customer will fill out a loan application and the manager will check the customer's info. If the customer chooses to pay in cash, the customer will need to bring the total amount to the dealership in order to complete the transaction.*

<details>
  <summary><b>Show output</b></summary>
  
<p align="center">
<img src="images/image_4.png" width="600">
</p>
</details>

---

### Example #5

*The process starts when the R&D team generates ideas for new products. At this point, 3 things occur in parallel: the first thing is the engineering team analyzing the ideas for feasibility. The engineering team also creates the technical specification. The second path involves the marketing team conducting market research for the ideas. At the same time, the design team creates visual concepts for the potential products. The third path sees the financial analysts reviewing the potential cost of the ideas. Once each track has completed its analysis, the management reviews the findings of the analysis.*

<details>
  <summary><b>Show output</b></summary>
  
<p align="center">
<img src="images/image_5.png" width="700">
</p>
</details>

---

### Example #6

*The company receives the order from the customer. If the product is out of stock, the customer receives a notification that the order cannot be fulfilled. If the product is in stock and the payment succeeds, the company processes and ships the order. If the product is in stock, but the payment fails, the customer receives a notification that the order cannot be processed.*

<details>
  <summary><b>Show output</b></summary>
  
<p align="center">
<img src="images/image_6.png" width="800">
</p>
</details>
