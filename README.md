# Process visualizer

This is an application that aims to convert textual descriptions of processes into simplified BPMN diagrams using [Graphviz](https://graphviz.org/).

Currently, the app supports the following BPMN elements:

* tasks
* exclusive gateways
* parallel gateways
* start and end events

Still under development and can only handle simple, structured inputs at this time.

Current limitations:

* Not able to handle loops (i.e. sequence flows returning to a previous activity)
* Maximum of 2 conditions per exclusive gateway
* Maximum of 2 paths per parallel gateway

## Example inputs and outputs

### Example #1

*The professor sends the mail to the student. After that, the student responds to the email. The
professor then reads the response. The student then waits for the answer.*

![image](images/image_1.png)

### Example #2

*The customer decides if he wants to finance or pay in cash. If the customer chooses to finance, the customer will need to fill out a loan application. After that, the customer sends the application to the bank. If the customer chooses to pay in cash, the customer will need to bring the total cost of \
the car to the dealership in order to complete the transaction. After the customer has chosen to finance or pay in cash, the customer must sign the contract before the transaction is completed.*

![image](images/image_2.png)

### Example #3

*The manager sends the mail to the supplier and prepares the documents. At the same time, the
customer searches for the goods and picks up the goods.*

![image](images/image_3.png)

### Example #4

*The customer decides if he wants to finance or pay in cash. If the customer chooses to finance, the
customer will need to fill out a loan application. Meanwhile, the manager checks the customer's info.
If the customer chooses to pay in cash, the customer will need to bring the total cost of the car to
the dealership in order to complete the transaction.*

![image](images/image_4.png)
