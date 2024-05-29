import csv
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_name = None
tfidf_vectorizer = TfidfVectorizer()


def preprocessing_user_input(text):
    if not text.strip():
        return None

    capitalized_words = re.findall(r"\b[A-Z\d&'][a-z\d:\s&'%d]*?\b", text)

    stop_words = set(stopwords.words('english'))
    capitalized_words = [word for word in capitalized_words if word.lower() not in stop_words]

    if not capitalized_words:
        return None

    result = ' '.join(capitalized_words)

    return result.strip('I').strip('i')


def set_movie_name(name):
    global movie_name
    movie_name = name


def get_movie_name():
    global movie_name
    return movie_name


def detect_num(user_input):
    numbers = re.findall(r'\d+', user_input)
    if numbers:
        return numbers[0]
    else:
        return None


# noinspection PyTypeChecker
def intent_enquiries(predicted_intent, user_input):
    if predicted_intent == "genre":
        genre = detect_genre(user_input)

        if genre:
            matching_movies = find_movies_by_genre(user_input)
            if matching_movies is not None:
                print(f"Bot: Here's are some movies based on {genre}")
                print(matching_movies)
            else:
                print(f"Bot: Sorry, no {genre.capitalize()} movies found.\n")
        else:
            genre = find_genre_by_movie(user_input)

            if genre:
                print(f"Bot: The genre of this movie is {genre}")

    elif predicted_intent == "cert_rating":
        cert_rating = detect_cert_rating(user_input)
        mov_name = detect_movie_name(user_input)

        if cert_rating:
            cert_matching_movies = find_movie_by_cert_rating(user_input)

            if cert_matching_movies:
                print(f"Bot: Here are movies that are Rated {cert_rating.capitalize()}:")
                for movie in cert_matching_movies:
                    print(movie)
            else:
                print(f"Bot: Sorry, no {cert_rating.capitalize()} rated movies found.")

        if mov_name:
            rated = find_cert_rating_movie(cert_rating)
            set_movie_name(mov_name)

            if rated:
                print(f"Bot: The movie is rated {rated}")
            else:
                print(f"Bot: Sorry, I could not find the rating for that movie.\n")
        else:
            if get_movie_name() is not None:
                rating_movie = find_cert_rating_movie(get_movie_name())
                if rating_movie:
                    print(f"Bot: The movie is rated {rating_movie}")
                else:
                    print(f"Bot: Sorry, I could not find a rating for that movie.")

    elif predicted_intent == "movie_info_summary":
        mov_name = detect_movie_name(user_input)
        if mov_name:
            matching_movies = find_summary_by_title(user_input)
            set_movie_name(mov_name)
            if matching_movies:
                print(f"Bot: Here's a summary for the movie {get_movie_name()}:")
                print(matching_movies)
            else:
                print(f"Bot: Sorry, I could not find a summary for that movie.\n")
        else:
            if get_movie_name():
                matching_movies = find_summary_by_title(user_input)
                if matching_movies:
                    print(f"Bot: Here's a summary for the movie {get_movie_name()}:")
                    print(matching_movies)
                else:
                    print(f"Bot: Sorry, I could not find a summary for that movie.\n")
            else:
                print("Bot: Sorry I could not find that movie :(")

    elif predicted_intent == "movie_info_rating":
        mov_name = detect_movie_name(user_input)

        if mov_name:
            set_movie_name(mov_name)  # Set the movie name
            rating_movie = find_rating_movie(user_input)

            if rating_movie:
                print(f"Bot: Here's the rating for the movie {get_movie_name()}")
                print(rating_movie)
            else:
                print(f"Bot: Sorry, I could not find a rating for that movie.")
        else:
            if get_movie_name() is not None:
                rating_movie = find_rating_movie(user_input)  # Use the retrieved movie name
                if rating_movie:
                    print(f"Bot: Here's the rating for the movie {get_movie_name()}")
                    print(rating_movie)
                else:
                    print(f"Bot: Sorry, I could not find a rating for the movie {get_movie_name()}.")
            else:
                print(f"Bot: I couldn't find the movie {get_movie_name()}.\n")

    elif predicted_intent == "movie_info_runtime":
        mov_name = detect_movie_name(user_input)

        if mov_name:
            runtime = find_runtime_movie(user_input)
            set_movie_name(mov_name)

            if runtime:
                print(f"Bot: Here's the runtime for the movie {get_movie_name()}:")
                print(runtime)
        else:
            if get_movie_name() is not None:
                runtime = find_runtime_movie(user_input)
                print(f"Bot: Here's the runtime for the movie {get_movie_name()}:")
                print(runtime)
            else:
                print(f"Bot: Sorry! I could not find the movie :(")

    elif predicted_intent == "movie_info_classification":
        mov_name = detect_movie_name(user_input)

        if mov_name:
            matching_movies = find_cert_rating_movie(user_input)
            set_movie_name(mov_name)

            if matching_movies:
                print(f"Bot: The movie is Rated: ")
                for cert in matching_movies:
                    print(cert)
            else:
                print(f"Bot: Sorry, I could not find the rating for that movie.\n")
        else:
            if get_movie_name() is not None:
                rating_movie = find_cert_rating_movie(user_input)
                if rating_movie:
                    print(f"Bot: The movie is Rated:")
                    for rating in rating_movie:
                        print(f"Bot: {rating}")
                else:
                    print(f"Bot: Sorry, I could not find a rating for that movie.")
            else:
                print("Bot: I couldn't find the movie.\n")

    elif predicted_intent == "movie_info_character":
        mov_name = detect_movie_name(user_input)

        if mov_name:
            matching_actors = find_actors_movie(user_input)
            set_movie_name(mov_name)

            if matching_actors:
                print(f"Bot: Here's the list of actors in the movie {get_movie_name()}")
                print(matching_actors)
            else:
                print(f"Bot: Sorry, I could not find the people who acted in that movie.\n")
        else:
            if get_movie_name() is not None:
                matching_actors = find_actors_movie(user_input)

                if matching_actors:
                    print(f"Bot: Here's the list of actors in the movie {get_movie_name()}")
                    print(matching_actors)
                else:
                    print(f"Bot: Sorry, I could not find the people who acted in that movie.\n")
            else:
                print("Bot: Sorry I could not find that movie")

    elif predicted_intent == "movie_info_director":
        director_name = detect_movie_name(user_input)

        if director_name:
            matching_name = find_directors_name(user_input)
            set_movie_name(director_name)

            if matching_name:
                print(f"Bot: Here's the list of directors in the movie {get_movie_name()}")
                print(matching_name)
            else:
                print(f"Bot: Sorry, I could not find the people who directed that movie.\n")
        else:
            if get_movie_name():
                matching_name = find_directors_name(user_input)

                if matching_name:
                    print(f"Bot: Here's the list of directors in the movie {get_movie_name()}")
                    print(matching_name)
                else:
                    print(f"Bot: Sorry, I could not find the people who directed that movie.\n")
            else:
                print("Bot: Sorry I could not find the movie")


def numbers_in_text(text):
    # Use regular expression to find the first numerical value in the text
    match = re.search(r'\d+', text)

    if match:
        return int(match.group())  # Return the first integer found
    else:
        return None


def detect_movie_name(user_input):
    mov_data = pd.read_csv('datasets/mov_data.csv')
    past_movie_list = pd.read_csv('datasets/past_movie_list.csv')

    preprocessed_input = preprocessing_user_input(user_input)

    if preprocessed_input is None:
        return None

    all_movie_titles = list(mov_data['title']) + list(past_movie_list['title'])

    all_movie_tfid = tfidf_vectorizer.fit_transform(all_movie_titles)

    user_input_tfid = tfidf_vectorizer.transform([preprocessed_input])

    mov_data_tfid = all_movie_tfid[:len(mov_data)]
    q_and_a_data_tfid = all_movie_tfid[len(mov_data):]

    cosine_similarities_mov_data = cosine_similarity(user_input_tfid, mov_data_tfid).flatten()
    cosine_similarities_q_and_a_data = cosine_similarity(user_input_tfid, q_and_a_data_tfid).flatten()

    max_similarity_mov_data = np.max(cosine_similarities_mov_data)
    max_similarity_q_and_a_data = np.max(cosine_similarities_q_and_a_data)

    if max_similarity_mov_data > max_similarity_q_and_a_data:
        most_similar_index = np.argmax(cosine_similarities_mov_data)
        closest_movie_name = mov_data.loc[most_similar_index, 'title']
    elif max_similarity_q_and_a_data > max_similarity_mov_data:
        most_similar_index = np.argmax(cosine_similarities_q_and_a_data)
        closest_movie_name = past_movie_list.loc[most_similar_index, 'title']
    else:
        return None

    capitalized_words = [word.strip(",") for word in closest_movie_name.split() if word[0].isupper()]
    result = ' '.join(capitalized_words)
    return result


def load_movie_data(csv_file):
    movie_data = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_data.append(row)
    return movie_data


def find_genre_by_movie(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    movie_data['genre'] = movie_data['genre'].str.strip()

    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    if detect_movie_name(user_input) is None:
        movie_name_vector = tfidf_vectorizer.transform([get_movie_name()])
    else:
        movie_name_vector = tfidf_vectorizer.transform([detect_movie_name(user_input)])

    cosine_similarities = cosine_similarity(movie_name_vector, movie_title_matrix)

    matching_index = (cosine_similarities > 0.7).flatten()

    matching_movies = list(movie_data['genre'].loc[matching_index])

    return str(matching_movies).strip("['']")


def find_movies_by_genre(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_data['genre'])
    user_input_vector = tfidf_vectorizer.transform([user_input])

    # If present in all, similarity_threshold is not needed
    if detect_genre(user_input) == "action" or detect_genre(user_input) == "adventure":
        threshold = 0
    else:
        threshold = 0.26

    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)

    matching_index = (cosine_similarities > threshold).flatten()

    matching_movies = list(movie_data['title'].loc[matching_index])

    return matching_movies


def find_summary_by_title(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()

    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    if detect_movie_name(user_input) is None:
        user_input_vector = tfidf_vectorizer.transform([get_movie_name()])
    else:
        user_input_vector = tfidf_vectorizer.transform([detect_movie_name(user_input)])

    cosine_similarities = cosine_similarity(user_input_vector, movie_title_matrix)

    most_similar_index = cosine_similarities.argmax()

    summary = movie_data['description'].iloc[most_similar_index]

    return summary


def detect_genre(user_input):
    # Tokenize the user input
    tokens = user_input.lower().split()

    # Define a list of possible genre keywords
    genre = [
        "animation",
        "comedy",
        "family",
        "adventure",
        "fantasy",
        "romance",
        "drama",
        "action",
        "crime",
        "thriller",
        "horror",
        "history",
        "sci-fi",
        "mystery",
        "documentary",
        "foreign",
        "music",
        "western",
        "war",
        "tv movie"
    ]

    # Find the genre mentioned in the user input
    for keyword in genre:
        if keyword in tokens:
            return keyword

    return None


def detect_cert_rating(user_input):
    tokens = user_input.lower().split()

    cert_rating = [
        "ua", "u", "r", "16+", "pg-13", "18+", "a", "pg"
    ]

    # Find the genre mentioned in the user input
    for keyword in cert_rating:
        if keyword in tokens:
            return keyword

    return None


def find_movie_by_cert_rating(user_input):  # Not Done
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()
    movie_cert_matrix = tfidf_vectorizer.fit_transform(movie_data['certificate'])

    if detect_cert_rating(user_input) == "u" or detect_cert_rating(user_input) == "r" or \
            detect_cert_rating(user_input) \
            == "a":
        matching_movies = []
        with open('datasets/mov_data.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['certificate'] == detect_cert_rating(user_input).upper():
                    matching_movies.append(row['title'])
        return matching_movies

    user_input_vector = tfidf_vectorizer.transform([detect_cert_rating(user_input)])

    cosine_similarities = cosine_similarity(user_input_vector, movie_cert_matrix)

    matching_index = (cosine_similarities == 1).flatten()

    matching_movies = list(movie_data['title'].loc[matching_index])

    return matching_movies


def find_runtime_movie(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()
    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    if detect_movie_name(user_input) is None:
        user_input_vector = tfidf_vectorizer.transform([get_movie_name()])
    else:
        user_input_vector = tfidf_vectorizer.transform([user_input])

    cosine_similarities = cosine_similarity(user_input_vector, movie_title_matrix)

    most_similar_index = cosine_similarities.argmax()

    runtime = movie_data['runtime'].iloc[most_similar_index]

    return runtime


def find_cert_rating_movie(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()
    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    if detect_movie_name(user_input) is None:
        user_input_vector = tfidf_vectorizer.transform([get_movie_name()])
    else:
        user_input_vector = tfidf_vectorizer.transform([user_input])

    cosine_similarities = cosine_similarity(user_input_vector, movie_title_matrix)

    matching_index = (cosine_similarities == 1).flatten()

    matching_movies = list(movie_data['certificate'].loc[matching_index])

    return ', '.join(matching_movies)


def find_rating_movie(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()
    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    user_input_vector = tfidf_vectorizer.transform([user_input])

    cosine_similarities = cosine_similarity(user_input_vector, movie_title_matrix)

    most_similar_index = cosine_similarities.argmax()

    rating = movie_data['rating'].iloc[most_similar_index]

    return rating


def find_actors_movie(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()
    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    if detect_movie_name(user_input) is None:
        user_input_vector = tfidf_vectorizer.transform([get_movie_name()])
    else:
        user_input_vector = tfidf_vectorizer.transform([user_input])

    cosine_similarities = cosine_similarity(user_input_vector, movie_title_matrix)

    most_similar_index = cosine_similarities.argmax()

    stars = movie_data['stars'].iloc[most_similar_index]

    return stars


def find_directors_name(user_input):
    movie_data = pd.read_csv('datasets/mov_data.csv')

    tfidf_vectorizer = TfidfVectorizer()
    movie_title_matrix = tfidf_vectorizer.fit_transform(movie_data['title'])

    if detect_movie_name(user_input) is None:
        user_input_vector = tfidf_vectorizer.transform([get_movie_name()])
    else:
        user_input_vector = tfidf_vectorizer.transform([user_input])

    cosine_similarities = cosine_similarity(user_input_vector, movie_title_matrix)

    most_similar_index = cosine_similarities.argmax()

    director = movie_data['director'].iloc[most_similar_index]

    return director
