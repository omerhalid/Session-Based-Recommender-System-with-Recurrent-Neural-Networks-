Retail Recommendation System with Recurrent Neural Networks

This repository contains the code for building a session-based recommendation system using PyTorch. The system is designed to recommend the next item to a user based on their session history.
Getting Started

To get started with this project, follow the instructions below.
Prerequisites

Make sure you have the following dependencies installed:

    Python (>= 3.6)
    NumPy
    pandas
    PyTorch (>= 1.0)
    scikit-learn

You can install these dependencies using pip:

bash

pip install numpy pandas torch scikit-learn

Dataset

The recommendation system is trained and tested on a dataset provided in a CSV file named "retailrocket_50percent.csv." This dataset contains retail user behavior data. Before running the code, make sure you have this dataset in the same directory as the project files.
Running the Code

The main code file is "recommendation_system.py." You can run the code by executing the following command:

  python recommendation_system.py

Project Structure

The project consists of the following components:

    Data Preprocessing: The dataset is loaded, and data preprocessing steps are performed, including filtering for "view" events and encoding item IDs and item types.

    SessionDataset Class: This custom PyTorch dataset class is defined to prepare the data for training and testing. It generates sequences of user sessions and their corresponding target items.

    SessionLSTM Model: A Long Short-Term Memory (LSTM) model is implemented using PyTorch. It takes session sequences as input and predicts the next item in the sequence.

    Training and Testing: The code includes code for training the model on a smaller sample of the dataset and evaluating its accuracy on a test set.

    Recommendation Function: A function is provided to recommend the next item based on a user's input session history.

Training and Testing

The model is trained for a single epoch due to the limited size of the dataset. In practice, you may want to train it for more epochs on a larger dataset to improve recommendation accuracy.
Example Usage

After running the code, you can interactively input a session history to get a recommendation for the next item.
Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request. We welcome any improvements, bug fixes, or additional features.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    This project is inspired by session-based recommendation systems.
    The dataset used in this project is sourced from the RetailRocket dataset.

Please feel free to reach out with any questions or feedback
