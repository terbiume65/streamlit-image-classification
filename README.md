# Simple General-Purpose Image Classifier using Streamlit
### Introduction
This is my second Streamlit project. The aim of the project is to allow users with no coding experience to create a Convolutional Neural Network(CNN) model and classify their own images. The app is based on a structure of pre-written, generalized template code for model training written in Tensorflow that leaves room to receive user input of training data and model hyperparameters. The dashboard gives users the freedom to choose how many classes of images they would like to classify (2-4), the complexity of the CNN network, training rate and optimizer, etc.
### Examples
In case you still can't conceptualize what image classification is, here are some of the common beginner examples that make use of image classification:
| Type of Classification                 | Objective                                                       |
| :---------------------                 | :--------------------                                             |
| `Binary Classification (2 classes)`    |Cats vs Dogs                            |
| `Binary Classification (2 classes)`    | Ship vs Cars                 |
| `Multiclass Classification (3+ classes)`       | Types of furniture                 |
| `Multiclass Classification (3+ classes)`       |Human emotions in images |
| `Multiclass Classification (3+ classes)`       | Types of food              |

Be creative, think of something useful and interesting to train a model with and predict!
### Credits
I encountered a lot of problems with the slightly complicated issue of caching in Streamlit's workflow. Streamlit re-runs and re-computes all variables whenever user input changes, which means the user's model creation process is often accidentally reset. Thanks to various discussion and answers offered in the incredibly friendly Streamlit community, I did not give up on the code and managed to finish this app, which is mostly working. Kudos to especially @_okld whose answers in the forum are very helpful. Some of his code, in the form of answers to questions raised in the Streamlit forum, have been used to solve some of my key issues and  finish this code.