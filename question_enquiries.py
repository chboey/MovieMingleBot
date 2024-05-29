import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie_enquiries import intent_enquiries, detect_movie_name, get_movie_name, set_movie_name
import random


def enquiries(intent, u_input):
    mov_data = pd.read_csv('datasets/mov_data.csv')
    q_and_a_data = pd.read_csv('datasets/q_and_a_data.csv')
    past_movie_data = pd.read_csv('datasets/past_movie_list.csv')

    tfidf_vectorizer = TfidfVectorizer()

    if detect_movie_name(u_input) is None and get_movie_name() is None:
        null_movie = input("Bot: Oh no! Looks like you were trying to find a query without specifying a\n"
                           "movie name :o Please let me know which movie you would like me to find"
                           " information about\nUser: ")
        null_movie_name = detect_movie_name(null_movie)
        set_movie_name(null_movie_name)
        if null_movie_name in past_movie_data['title'].values:
            print("Bot: Please enter a movie that is currently screening.")
            return
        mov_name_and_data_tfid = tfidf_vectorizer.fit_transform(
            [null_movie_name] + list(mov_data['title']))
    else:
        if get_movie_name() is None:
            detected_movie_name = detect_movie_name(u_input)
            set_movie_name(detected_movie_name)
            mov_name_and_data_tfid = tfidf_vectorizer.fit_transform([detected_movie_name] + list(mov_data['title']))
        else:
            mov_name_and_data_tfid = tfidf_vectorizer.fit_transform([get_movie_name()] + list(mov_data['title']))

    user_input_tfid = tfidf_vectorizer.fit_transform([u_input])
    past_mov_tfid = tfidf_vectorizer.transform(q_and_a_data['question'])

    mov_name_tfid = mov_name_and_data_tfid[:1]
    mov_data_tfid = mov_name_and_data_tfid[1:]

    similarity_mov_data = cosine_similarity(mov_name_tfid, mov_data_tfid)
    similarity_q_and_a = cosine_similarity(user_input_tfid, past_mov_tfid)
    similarity_user_input = cosine_similarity(user_input_tfid, past_mov_tfid)

    max_mov_data = np.max(similarity_mov_data)
    max_q_and_a_data = np.max(similarity_q_and_a)

    if max_mov_data > max_q_and_a_data:
        intent_enquiries(intent, u_input)

        random_number = random.randint(0, 3)

        if random_number == 0:
            print(" ")
            print("----------------------Bot Hints---------------------------")
            print("Bot: Here's a hint, You can book tickets for this movie :D")

    else:
        max_index = np.argmax(similarity_user_input)

        answer = q_and_a_data.loc[max_index, 'answer']

        print(f"Bot: Here you go!\nBot:{answer}")
