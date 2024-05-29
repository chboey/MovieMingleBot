import csv
import hashlib
import logging
import re
from datetime import datetime

import nltk
# import numpy as np [ Uncomment this for Evaluation of Intent Detection ]
# import pandas as pd [ Uncomment this for Evaluation of Intent Detection ]
# import seaborn as sns [ Uncomment this for Evaluation of Intent Detection ]
# from matplotlib import pyplot as plt [ Uncomment this for Evaluation of Intent Detection ]
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from question_enquiries import enquiries
from small_talk import bot_enquiries
from transactions import mov_transactions
from name_mgmt import detect_names, register_user, login_user, custom_tokenizer, register_name, get_name_csv, \
    login_getname, change_name, set_name_mgmt, get_name_mgmt
from training_data.training_data import training_data

# from sklearn.model_selection import train_test_split [ Uncomment this for Evaluation of Intent Detection ] from
# sklearn.linear_model import LogisticRegression [ Uncomment this for Evaluation of Intent Detection ] from
# sklearn.pipeline import make_pipeline [ Uncomment this for Evaluation of Intent Detection ] from sklearn.metrics
# import accuracy_score, confusion_matrix, classification_report [ Uncomment this for Evaluation of Intent Detection ]


# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


def preprocessing_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = custom_tokenizer.tokenize(text.lower())
    pos_tag = nltk.pos_tag(tokens)

    lemmatized_tokens = [lemmatizer.lemmatize(word, pos=pos_map(pos))
                         for word, pos in pos_tag if word not in set(stopwords.words('english'))]

    return ' '.join(lemmatized_tokens)


def pos_map(treebank_tag):
    return {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}.get(treebank_tag[0], 'n')


def find_closest_intent(u_input, data):
    user_vector = vectorizer.transform([preprocessing_text(u_input)])
    max_similarity = 0
    closest_intent = None

    for text, intent in data:
        training_data_vector = vectorizer.transform([preprocessing_text(text)])
        similarity = cosine_similarity(user_vector, training_data_vector)[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            set_threshold(max_similarity)
            closest_intent = intent

    return closest_intent


def prompts(count, name_prov, count2):
    if count == 1 and counter2 == 0:
        start_template = """
            -------------------------------------------------------------------------------------------------
                                    Welcome to MovieMingleBot!

                    MovieMingleBot is your ultimate movie companion, offering a variety of services:

                    1) Booking or cancel tickets for the latest blockbusters.
                    2) Get showtimes for movies that are currently screening.
                    3) Inquire about both past and current movies.
                    4) Engage in casual chats.

                    To unlock the full range of services, don't forget to register or log in!
            -------------------------------------------------------------------------------------------------
                    """

        print(start_template)

    elif count2 == 1 and name_prov:
        welcome_template = """
            -------------------------------------------------------------------------------------------------
                    I can get you about information about movies, both currently screening and
                    past releases. Here's how you can ask about me questions:

                    - To inquire about currently screening movies:
                        Example: "what movies are currently screening right now?"

                    - To inquire about movie information:
                        Example: "what is the summary for the movie Deadpool?
                        Example: "who acted in this movie?"
                        Example: "what is the runtime for this movie" (After enquiring about a movie that is screening)
                        Example: "what is this movie rated for?"
                        Example: "what is the rating fort his movie?"
                        Example: "who directed this movie"

                    - To book movie tickets:
                        Example: "book tickets for the movie Deadpool" or
                        Example: "book tickets for this movie" (After enquiring about it)

                    - To cancel movie ticket:
                        Example: "cancel my movie tickets"
                        Example: "I don't feel like watching this movie anymore"

                    - To get details about past movies:
                        Example: "what movies were screened in this cinema?"

                    - To change what you would like me to call you:
                        Example: "I would like you to call me Bob instead of Boey"

                    - To get weather details / ask for time / etc.:
                        Example: "What is the weather like today/tomorrow/in a few days"
                        Example: "What time is it right now?"
            -------------------------------------------------------------------------------------------------
            """
        print(welcome_template)

    else:
        print()


training_data_preprocessed = [preprocessing_text(text) for text, intent in training_data]

vectorizer = TfidfVectorizer()
training_vector = vectorizer.fit_transform(training_data_preprocessed)

# Logistic Regression to determine the performance of the intent detection system

# X = training_data_preprocessed
# y = [intent for text, intent in training_data]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# classifier = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200000, solver='sag',
#                                                                 random_state=62, C=20))
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# print(f"Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)
# misclassified_classes = np.sum(confusion_matrix) - np.trace(confusion_matrix)
# correctly_classified_classes = np.trace(confusion_matrix)
# print(f"Number of correctly classified classes: {correctly_classified_classes}")
# print(f"Number of misclassified classes: {misclassified_classes}")

# Plot the confusion matrix as a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
#            xticklabels=[f"Class {i}" for i in range(cm.shape[0])],
#            yticklabels=[f"Class {i}" for i in range(cm.shape[0])])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()
# print(f"Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

user_input = ""
user_name = ""

name_provided = False
logged_in = False

counter = 0
counter2 = 0
threshold = 0

interaction_history = []


def set_name_main(name):
    global user_name
    user_name = name


def get_name_main():
    return user_name


def set_threshold(thresh):
    global threshold
    threshold = thresh


def get_threshold():
    return threshold


while True:
    if not logged_in:

        counter += 1
        prompts(counter, None, 0)
        user_input = input("User: ")

        # start_time = time.time()

        predicted_intent = find_closest_intent(user_input, training_data)

        # end_time = time.time()

        # response_time = end_time - start_time

        # print(f"Bot: Response time: {response_time:.2f} seconds")

        if user_input and predicted_intent == "name_mgmt" and re.search(r'\b(?:name|name?|Hi|Hello|Heya)\b', user_input,
                                                                        re.IGNORECASE):
            if not name_provided:
                saved_name = detect_names(user_input)
                if saved_name is not None:
                    register_name(saved_name)
                    name_provided = True
                    set_name_main(saved_name)
                    print(f"Bot: Hello {saved_name}!, to use our services, make sure that you have an account with us. "
                          f"Login if you have, otherwise Register please!")
                else:
                    print("Bot: Sorry! No name found")

            else:
                with open('datasets/user_data.csv', 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                print(f"Bot: You have already given me your name, I've remembered that it is.. {get_name_csv(rows)}!")

        elif predicted_intent == "name_mgmt" and re.search(r'\b(?:name|name?)\b', user_input, re.IGNORECASE):
            if not name_provided:
                give_name = input("Bot: You have not given me your name yet. Can I have it, please? ")
                detected_name = detect_names(give_name)

                if detected_name is not None and len(detected_name) > 0:
                    register_name(detected_name)
                    name_provided = True
                    set_name_main(detected_name)
                    print(f"Bot: Alright {detected_name}! I will remember it from now on!")
                    break
                else:
                    print("Bot: No valid name detected, try again!")

            else:
                with open('datasets/user_data.csv', 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                print(f"Bot: Your name is {get_name_csv(rows)} :D")

        if predicted_intent == "register":
            if not name_provided:
                give_name = input("Bot: You need to give me your name before you register! \nUser: ")
                given_name = detect_names(give_name)
                if given_name is None:
                    print("Bot: No name was found, please try again!")
                    name_provided = False
                else:
                    set_name_main(given_name or get_name_main())
                    register_name(given_name or get_name_main())
                    name_provided = True
                    print(f"Bot: Thank you {given_name or get_name_main()}! You may now proceed to register :)")
            else:
                username = input("Enter a username: ")
                password = input("Enter your password: ")
                register_user(username, password)

        elif predicted_intent == "login":
            username = input("Enter your username: ")
            password = input("Enter your password: ")
            print("\n")
            if login_user(username, password):
                with open('datasets/user_data.csv', 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                    user_name = login_getname(username, hashlib.sha256(password.encode()).hexdigest())
                    set_name_main(user_name)
                    logged_in = True
                    print(f"Bot: Hi {user_name}! You are now logged in.")
            else:
                print("Bot: Login failed. Invalid username or password.")

        elif predicted_intent == "exit":
            if get_name_main():
                print(f"Bot: Goodbye! {get_name_main()}")
            else:
                print("Bot: Goodbye!")
            break

        elif predicted_intent == "change_name":
            if not name_provided:
                user_input_name = input("Bot: You have not given me your name.. Can I have it please?\nUser:")
                saved_name = detect_names(user_input_name)
                set_name_main(saved_name)
                register_name(saved_name)
                print(f"Bot: Alright! I will remember you as {saved_name}")
                continue
            else:
                new_name = input("Bot: Roger that! What would like me to call you?\nUser:")
                saved_new_name = detect_names(new_name)
                set_name_main(saved_new_name)
                change_name(new_name, logged_in=False)
                print(f"Bot: Okay! I will remember you as {saved_new_name}")

        elif predicted_intent == "exit":
            print("You are not logged in yet!")

        elif predicted_intent == "swear_words":
            logging.basicConfig(filename='chatlogs/chatlog.txt', level=logging.INFO)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            if get_name_main():
                logging.info(f'{timestamp}({(get_name_main())}) -> {user_input}')
            else:
                logging.info(f'{timestamp}(Anon) -> {user_input}')
            print("Bot: Swear words/ racial slurs would not be tolerated by this service. >:(")
            break

        elif predicted_intent is None:
            print(
                "I'm sorry, I couldn't understand your request. Please try again or choose a valid option (Login or "
                "Register).")

    else:
        with open('datasets/user_data.csv', 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            name_provided = True
            counter2 += 1
            prompts(counter, name_provided, counter2)
            user_input = input("User: ")

            # start_time = time.time()

            predicted_intent = find_closest_intent(user_input, training_data)

            # end_time = time.time()

            # response_time = end_time - start_time

            # print(f"Bot: Response time: {response_time:.2f} seconds")

        if predicted_intent == "exit":
            print(f"Goodbye, {get_name_mgmt()}!")
            user_name = None
            break

        if predicted_intent == "swear_words":
            logging.basicConfig(filename='chatlogs/chatlog.txt', level=logging.INFO)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            logging.info(f'{timestamp}({(get_name_main())}) -> {user_input}')
            print("Bot: Swear words/ racial slurs would not be tolerated by this service. >:(")
            break


        def process_intent(intent):
            if intent in ["genre", "cert_rating", "movie_info_summary", "movie_info_popularity",
                          "movie_info_rating", "movie_info_runtime", "movie_info_classification",
                          "movie_info_character", "movie_info_director"]:
                enquiries(intent, user_input)

            elif intent == "ticket_booking":
                mov_transactions(intent, user_input)

            elif intent in ["transaction_info", "movie_showtimes", "movie_seat_availability",
                            "movie_screening", "time", "ticket_cancelling", "show_bookings", "movie_info_past"]:
                mov_transactions(intent, user_input)

            elif intent in ["bot_enquiries", "bot_enquiries_weather", "bot_enquiries_time",
                            "bot_enquiries_weather",
                            "bot_enquiries_sentimental"]:
                bot_enquiries(intent, user_input)

            elif intent == "change_name":
                logged_in_new_name = input("Bot: Sure! What would like me to call you?\nUser:")
                saved_logged_in_new_name = detect_names(logged_in_new_name)
                if saved_logged_in_new_name:
                    change_name(saved_logged_in_new_name, logged_in=True)
                    set_name_mgmt(saved_logged_in_new_name)
                    print(f"Bot: Okay! I will remember you as {saved_logged_in_new_name}")
                else:
                    print(f"Bot: No name was detected, please try again :(")

            elif intent == "name_mgmt" and re.search(r'\b(?:name|name?)\b', user_input, re.IGNORECASE):
                if get_name_mgmt() is not None:
                    print("Bot: I know you're trying to test my memory but... I remember that "
                          f"your name is...{get_name_mgmt()}")


        if predicted_intent is not None:
            interaction_history.append((user_input, predicted_intent))

            if predicted_intent == "back":
                if len(interaction_history) >= 3 and interaction_history[-3] is not None:
                    prev_enquiry = find_closest_intent(interaction_history[-3][0], training_data)

                    if prev_enquiry != predicted_intent:  # Check if going back to the same intent
                        process_intent(prev_enquiry)

                        if prev_enquiry == "ticket_cancelling":
                            print("Bot: Ticket cancellation cannot be undone. So sorry!")

                    else:
                        print("Bot: Already at the previous enquiry.")
                else:
                    print("Bot: Type logout/exit to leave the chat :D")
            else:
                process_intent(predicted_intent)

            if predicted_intent == "gratitude":

                if len(interaction_history) >= 3:
                    prev_user_input, prev_predicted_intent = interaction_history[-2]

                    data_entry = (prev_user_input, prev_predicted_intent)
                    training_data.append(data_entry)

                    updated_content = f"training_data = [\n"
                    for entry in training_data:
                        updated_content += f"    {repr(entry)},\n"
                    updated_content += "]\n"

                    with open('training_data/training_data.py', 'w') as file:
                        file.write(updated_content)

                    print("Bot: Thanks for your feedback!")

                else:
                    print("Bot: I don't think I have really helped you out alot... but You're Welcome!!")

            if predicted_intent == "login":
                print("Bot: You are already logged in :D")

            if predicted_intent == "register":
                print("Bot: You have already registered as well....")

            if get_threshold() < 0.75:
                if predicted_intent is not None:
                    user_satisfaction = input("Bot: Are you satisfied with my response? (yes/no) ").lower()

                    if user_satisfaction == 'yes':
                        data_entry = (user_input, predicted_intent)
                        training_data.append(data_entry)

                        updated_content = f"training_data = [\n"
                        for entry in training_data:
                            updated_content += f"    {repr(entry)},\n"
                        updated_content += "]\n"

                        with open('training_data/training_data.py', 'w') as file:
                            file.write(updated_content)

                        print("Bot: Thanks for your feedback!")
                    else:
                        print("Bot: Thank you for your feedback.")

            if predicted_intent == "bot_capabilities":
                print("Bot: I am able to get you information about movies / book tickets / cancel tickets")

            if predicted_intent == "help":
                template = """
                            -------------------------------------------------------------------------------------------------
                                    I can get you about information about movies, both currently screening and
                                    past releases. Here's how you can ask about me questions:

                                    - To inquire about currently screening movies:
                                        Example: "what movies are currently screening right now?"

                                    - To inquire about movie information:
                                        Example: "what is the summary for the movie Deadpool?
                                        Example: "who acted in this movie?"
                                        Example: "what is the runtime for this movie" (After enquiring about a movie 
                                                                                                    that is screening)
                                        Example: "what is this movie rated for?"
                                        Example: "what is the rating fort his movie?"
                                        Example: "who directed this movie"

                                    - To book movie tickets:
                                        Example: "book tickets for the movie Deadpool" or
                                        Example: "book tickets for this movie" (After enquiring about it)

                                    - To cancel movie ticket:
                                        Example: "cancel my movie tickets"
                                        Example: "I don't feel like watching this movie anymore"

                                    - To get details about past movies:
                                        Example: "which movies were screening in the past?"

                                    - To change what you would like me to call you:
                                        Example: "Change my name please!"

                                    - To get weather details / ask for time / etc.:
                                        Example: "What is the weather like today/tomorrow/in a few days"
                                        Example: "What time is it right now?"
                                    
                                    -If you are lost:
                                        Example: "Help"
                                        Example: "What state am I currently in?"
                                    
                                    -If you want to go back:
                                        Example: "go back:
                                        
                            -------------------------------------------------------------------------------------------------
                            """
                print(template)

            if predicted_intent == "rewind":
                if len(interaction_history) >= 2:
                    prev_user_input, prev_predicted_intent = interaction_history[-2]
                    intent_key = {
                        "movie_showtimes": "You were enquiring about movie showtimes",
                        "movie_screening": "You were enquiring about movie screening times",
                        "movie_seat_availability": "You were enquiring about available seats for a movie",
                        "ticket_booking": "You were in the process of booking tickets",
                        "ticket_cancelling": "You were cancelling your booking :(",
                        "show_booking": "You were enquiring about your bookings with us",
                        "movie_info_summary": "You were enquiring about the summary of a movie",
                        "movie_info_popularity": "You were enquiring about popular movies",
                        "movie_info_runtime": "You were enquiring about the runtime of a movie",
                        "movie_info_rating": "You were enquiring about the rating of a movie",
                        "movie_info_classification": "You were enquiring about the classification of a movie",
                        "movie_info_character": "You were enquiring about the characters in a movie",
                        "name_mgmt": "You were managing your name",
                        "login": "You were logging in",
                        "genre": "You were enquiring about movie genres",
                        "cert_rating": "You were enquiring about movie certification ratings",
                        "bot_enquiries": "You were making general inquiries to the bot",
                        "bot_enquiries_weather": "You were enquiring about the weather through the bot",
                        "bot_enquiries_time": "You were enquiring about the time through the bot",
                        "bot_enquiries_sentimental": "You were inquiring about sentimental analysis through the bot",
                        "gratitude": "You expressed gratitude",
                        "bot_capabilities": "You were exploring the capabilities of the bot",
                        "rewind": "You wanted to go back or rewind the conversation"
                    }
                    if prev_predicted_intent in intent_key:
                        bot_response = intent_key[prev_predicted_intent]
                        print(f"Bot: {bot_response}")
                    else:
                        print("Bot: I don't have information about the previous intent.")
                else:
                    print("Bot: Uhh... I don't think you are doing anything right now... :o")

        else:
            print("Bot: Sorry, I could not help you with this :(, you could type 'help' for instructions")
