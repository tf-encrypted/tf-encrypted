# Private Prediction and Training of Logistic Regression Model

The examples are as follows:

- [`prediction_single.py`](./prediction_single.py) shows how a single client can run a private prediction
- [`prediction_joint.py`](./prediction_joint.py) shows how several clients can combine features for a prediction
- [`training_single.py`](./training_single.py) shows how a model can be privately trained on data from a single owner
- [`training_joint.py`](./training_joint.py) shows how several data owners can jointly train on their combined data

Note that the training data in the above examples are randomly sampled. [Here](https://alibaba-gemini-lab.github.io/docs/blog/tfe/) gives a more realistic example showing two data owners jointly training on vertically split dataset, which are separately read from their local csv files.
