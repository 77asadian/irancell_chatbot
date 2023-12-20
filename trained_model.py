# import the libraries.
import pandas as pd
import torch
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer
import numpy as np

# Define a function to answer the question based on the pre-defined dataset and embeddings.
def answer_to_question(question):
    question = [question]                                                               # turn question to a list for preprocessing.
    answers = pd.read_csv('dataset_english.csv')['1'].values                            # reading the answers from dataset
    questions = pd.read_csv('dataset_english.csv')['0'].values                          # reading the questions from the dataset
    model_id = "sentence-transformers/all-MiniLM-L6-v2"                                 # choosing a proper model
    model = SentenceTransformer(model_id)                                               # creating the model
    embeddings = pd.read_csv('embeddings_english.csv')                                  # reading the embedding vectors
    dataset_embeddings = torch.from_numpy(embeddings.to_numpy()).to(torch.float)        # turning embeddings into torch types
    output = model.encode(question)                                                     # encode the question of the user
    output_embeddings = torch.FloatTensor(output)                                       # turning embedding into torch type
    hits = semantic_search(output_embeddings, dataset_embeddings, top_k=4)              # getting the proper answers
    my_answers = []                                                                     # list of proper answers to the question
    pre_score = hits[0][0]['score'] + 0.001                                             # initialize the pre_score
  
    for answer_data in hits[0]:
        answer_score = answer_data['score']         # storing the firts hit's score
        # the respones should not be the same. responses should be proper to the question. if two responses are similar, we return them both.
        if (answer_score != pre_score) and (answer_score > 0.45) and ((pre_score - answer_score < 0.05) or ((answer_score > 0.7) and (pre_score - answer_score < 0.05))):
            my_answers.append(answers[answer_data['corpus_id']])
            pre_score = answer_score
        else:
            break       # In case we couldn't find anything proper from the dataset.

    output = ""         # making the output. a string of the answers.
    if len(my_answers) > 0 :
        for answer in my_answers:
          output += answer + " "
        return output
    else:
        sug_quest = questions[hits[0][0]['corpus_id']]
        return f"The question is unclear! Here is a suggested question: {sug_quest}"       # if there was not a proper answer, we suggest the closest question.