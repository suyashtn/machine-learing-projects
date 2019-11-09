# Machine Learning, Deployment Case Studies with AWS SageMaker

This repository contains code and associated files for deploying ML models using AWS SageMaker. This repository consists of a number of projects that will be used to illustrate parts of the ML workflow and can be used to practice deploying a variety of ML algorithms.

### Projects

[Sentiment Analysis Web App](https://github.com/suyashtn/machine-learing-projects/tree/master/Deploy_Sentiment_Analysis_Model): is a deployed RNN performing sentiment analysis on movie reviews complete with publicly accessible API and a simple web page which interacts with the deployed endpoint. This project assumes that you have some familiarity with SageMaker and used the built-in XGBoost model. Find original project files in [Udacity repository](https://github.com/udacity/sagemaker-deployment/tree/master/Project).


[Plagiarism Detector](https://github.com/suyashtn/machine-learing-projects/tree/master/Project_Plagiarism_Detection): Build an end-to-end plagiarism classification model. Apply your skills to clean data, extract meaningful features, and deploy a plagiarism classifier in SageMaker. Find original project files in [Udacity repository](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Project_Plagiarism_Detection).


[Stock Pirce Predictor](https://github.com/suyashtn/machine-learing-projects/tree/master/Predict_Stock_Prices): Build a stock price prediction model. The share price datasets for publicly listed companies can be obtained from [Quandl](https://www.quandl.com). In this project, the Linear Regression and XGBoost methods are used as benchmark models and the performance of an implementation of the Long Short-Term Memory (LSTM) network model is compared for making future predictions.

---

## Setup Instructions

The notebooks provided in this repository are intended to be executed using Amazon's SageMaker platform. The following is a brief set of instructions on setting up a managed notebook instance using SageMaker, from which the notebooks can be completed and run.

### Log in to the AWS console and create a notebook instance

Log in to the [AWS console](https://console.aws.amazon.com) and go to the SageMaker dashboard. Click on 'Create notebook instance'.
* The notebook name can be anything and using `ml.t2.medium` is a good idea as it is covered under the free tier.
* For the `role`, creating a new role works fine. Using the default options is also okay.
* It's important to note that you need the notebook instance to have access to `S3` resources, which it does by default. In particular, any S3 bucket or object, with “sagemaker" in the name, is available to the notebook.
* Use the option to **git clone** the project repository into the notebook instance by pasting `https://github.com/suyashtn/machine-learing-projects/`

### Open and run the notebook of your choice

Now that the repository has been cloned into the notebook instance you may navigate to any of the notebooks that you wish to complete or execute and work with them. Additional instructions are contained in their respective notebooks.
