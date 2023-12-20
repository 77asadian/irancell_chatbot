# Hackathon Challenge - Irancell Second Place Winner

## Overview

This repository contains the code that secured the second place in the Irancell hackathon challenge. The challenge involved developing a model capable of answering questions based on a pre-defined dataset and embeddings. The model includes a user interface (UI) built with the Streamlit library.

## Requirements

Make sure you have the following libraries installed before running the code:

- pandas
- torch
- sentence-transformers

You can install them using the following command:

```bash
pip install pandas torch sentence-transformers
```

## Code Structure

The main code file (`trained_model.py`) imports the necessary libraries and defines a function (`answer_to_question`) to generate responses based on the pre-defined dataset and embeddings.

- `dataset_english.csv`: Contains questions and answers for training the model.
- `embeddings_english.csv`: Stores embedding vectors used by the model.
- `trained_model.py`: The main script containing the code for the question-answering model.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/77asadian/irancell_chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd project
   ```

3. Run the code:

   ```bash
   python streamlit run app.py
   ```

   The code initializes the model, processes user input, and provides answers based on the pre-trained embeddings.

## Model Details

The model uses the "sentence-transformers/all-MiniLM-L6-v2" model for question encoding and semantic search. It filters and selects proper answers based on similarity scores.

## Results

The model achieved second place in the Irancell hackathon challenge by effectively answering questions from the given dataset.

## Contact

For any inquiries or collaborations, feel free to contact the repository owner:

- Hamidreza Asadian
- 77asadian@gmail.com
- [My Linkedin)](https://www.linkedin.com/in/77asadian/)https://www.linkedin.com/in/77asadian/
