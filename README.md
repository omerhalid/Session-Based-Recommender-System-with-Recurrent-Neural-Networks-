# Retail Recommendation System with Recurrent Neural Networks

## Overview
This repository hosts a session-based recommendation system implemented using PyTorch. It's tailored to recommend the next item in a shopping session based on user behavior history.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python (>= 3.6)
- NumPy
- pandas
- PyTorch (>= 1.0)
- scikit-learn

Install these using pip:
```bash
pip install numpy pandas torch scikit-learn
```

##Dataset

The system utilizes the "retailrocket_50percent.csv" dataset, containing user behavior data in retail. Place this dataset in the project directory before execution.

## Running the Code

Execute the main script to start the recommendation system:

```bash
python recommendation_system.py
```

## Project Structure

    Data Preprocessing: Processes the dataset, including filtering and encoding.
    SessionDataset Class: A PyTorch class for preparing training and testing data.
    SessionLSTM Model: LSTM model for predicting the next item in a session.
    Training and Testing: Scripts for model training and accuracy evaluation.
    Recommendation Function: Generates item recommendations based on user sessions.

## Training and Testing

The model is initially trained for a single epoch on a subset of the dataset. For improved accuracy, consider more epochs and a larger dataset.
#Example Usage

Input a user session history to receive item recommendations.

## Contributing

Contributions are welcome! Fork the repo and submit pull requests for any improvements or bug fixes.
License

Licensed under the MIT License. See LICENSE for details.
Acknowledgments

    Inspired by session-based recommendation systems.
    Dataset sourced from RetailRocket.
