# **Welcome to My Machine Learning Examples Repository!**

This repository is your destination for exploring a wide array of machine learning use cases. From beginner-friendly notebooks to advanced notebooks, my collection is curated to cater to enthusiasts and professionals alike. Whether you're interested in computer vision, predictive analytics, or any other ML domain, you'll find practical examples that not only demonstrate the theoretical aspects but also provide hands-on experience with real-world datasets. I include a python utilities file to make plotting, statistical analysis and model metric reporting simple to compliment the noteboks.

## **Dive into my examples**

***Discover:*** Explore diverse applications of machine learning across different industries and domains.

***Learn:*** Gain insights into various algorithms and techniques, from the basics to cutting-edge developments.

***Apply:*** Implement machine learning models on your projects with our easy-to-follow code examples.

*The datasets my examples utilize are from free online sources. I do not store datasets in my repo. The notebook will have a proper working link to import the datasets.* 

***My notebooks follow the same flow as depicted below:***

![Notebook ML Steps](./images/mlProcess.png "Flow")

My repository is continuously updated with new examples, ensuring you're always equipped with several examples. 


| Example | Level | Category | Model/s Used | Short Summary | Link |
|---------|-------|----------|--------------|---------------|------|
| Sales Predictions | Simple | Regression | Oridinary Least Squares | Predicting the Sales for the next qualrter | [Link](Regression-Sales-Prediction)|
| Hospital Length of Stay Predictions | Moderate | Regression | Decision Tree, Random Forest, XG Boost | Predicting how long someone will stay at the hospital | [Link](Regression-Hospital-LOS)|
| Customer Conversion Prediction | Moderate| Classification | Decision Tree, Random Forest, AdaBoost, Gradient Boost | Predicting potential lead conversion to paid customers | [Link](Classification-Customer-Conversion-Prediction) |
| Anomaly Detection | Moderate | Deep Learning | Autoencoder (FNN) | Detecting anomalous heart rythms | [Link](AnomalyDetection-HeartECG)
| Loan Default Prediction | Moderate | Classification | Decision Tree, Random Forest, AdaBoost, Gradient Boost, LDA, QDA, Logistic Rrgression, SVM, KNN, NB | Predicting Loan Defaults | [Link](Classification-Loan-Default-Prediction)


## **Pre-Requisites and Install**

Your personal laptop/desktop will need to have Python installed and we highly recommend using Python 3.10. You can use a tool like pyenv (mac) or pyenv-win (windows) to easily download and switch between Python versions. (I am running a Mac M1, so the commands below are for that hardware.)

***Install python via pyenv and setting the version globally***

- `pyenv install 3.10.11`  # install
- `pyenv global 3.10.11`  # set default

***Once we have our Python version, we can create a directory, clone the repository and create a virtual environment to install our dependencies.***

1. Make a directory.
2. Clone this repository to that directory. 
3. Create a virtual environment by performing the below commands:
   - `python3 -m venv mlvenv`  # create virtual environment
   - `source mlvenv/bin/activate`  # on Windows: venv\Scripts\activate
4. Install the dependencies by performing the following commands:
   - `python3 -m pip install --upgrade pip setuptools wheel`
5. Install requirements from the requirements.txt file.
   - `pip install -r requirements.txt`

**Notes on TensorFlow**

I ran this on a Mac M1 Pro, python 3.10.11.
The `requirements.txt` file does not contain `tensorflow`. 

For TensorFlow, I installed the standard package through: `pip install tensorflow`. And becuase I am on a Mac with a M1 chip, I needed to install `tensorflow-metal` to take advantage of the GPU.I installed it with `pip install tensorflow-metal` ***OR*** you can use the extended requirements_with_tf.txt file bu using: `pip install -r requirements_with_tf.txt`

For your specific local machine, if it does not have a GPU, the models will still work with CPU, but just take longer.


## ***Enjoy!***