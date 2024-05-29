import csv
import hashlib
import os
import nltk
from nltk import RegexpTokenizer

# CSV file for user data (username, hashed password)
USER_DATA_FILE = 'datasets/user_data.csv'


# Function to create the user data file if it doesn't exist
def create_user_data_file():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['username', 'password'])


# Call the function to create the user data file
create_user_data_file()
custom_tokenizer = RegexpTokenizer(r'\w+')

name = []
username = []
password = []


def set_name_mgmt(current_name):
    global name
    name = current_name


def get_name_mgmt():
    return name


def set_username(stored_username):
    global username
    username = stored_username


def get_username():
    return username


def set_hashed_password(stored_password):
    global password
    password = stored_password


def get_hashed_password():
    return password


def detect_names(user_input):
    if not isinstance(user_input, str):
        return None

    sentences = nltk.sent_tokenize(user_input)

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        named_entities = nltk.ne_chunk(pos_tags)

        for subtree in named_entities:
            if type(subtree) == nltk.Tree and subtree.label() in ['GPE', 'ORGANIZATION', 'PERSON', 'NP', 'NNP']:

                return " ".join(word for word, tag in subtree.leaves())

    return None


def register_name(user_input):
    if user_input:
        with open(USER_DATA_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([user_input])


def register_user(user, pw):
    with open(USER_DATA_FILE, mode='r', encoding='utf-8') as file:
        # Read the existing data as a list of dictionaries
        reader = csv.DictReader(file)
        rows = list(reader)

    # Check if the username is already taken
    for row in rows:
        if row['username'] == user:
            print("Username is already taken. Please choose a different one.")
            return

    # Hash the password before storing it
    hashed_password = hashlib.sha256(pw.encode()).hexdigest()

    # Find the first row that has an empty column (e.g., 'name') and update it
    for row in rows:
        if not row.get('username'):
            row['username'] = user
            row['password'] = hashed_password
            break
    else:
        # If no row with an empty column is found, create a new row
        new_row = {'username': user, 'password': hashed_password}
        rows.append(new_row)

    # Write the updated data back to the CSV file
    with open(USER_DATA_FILE, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['name', 'username', 'password', 'seats_booked', 'time_booked', 'movie_booked', 'num_of_seats']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Registration successful. You can now log in.")


def login_user(user, pw):
    # Hash the entered password to match against stored hashed passwords
    hashed_password = hashlib.sha256(pw.encode()).hexdigest()

    with open(USER_DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3 and row[1] == user and row[2] == hashed_password:
                return True
    return False


def get_name_csv(rows):
    global c1
    for row in rows:
        if row:
            c1 = row[0]
    return c1


def login_getname(user, pw):
    with open('datasets/user_data.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists
        for col in reader:
            name, stored_username, stored_hashed_password = col[:3]
            if stored_username == user and stored_hashed_password == pw:
                set_name_mgmt(name)
                set_username(stored_username)
                set_hashed_password(stored_hashed_password)
                return name
    return None


def change_name(user_input, logged_in):
    new_name = detect_names(user_input)

    if logged_in:
        current_name = get_name_mgmt()
        set_name_mgmt(current_name)
    else:
        with open('datasets/user_data.csv', 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        current_name = get_name_csv(rows)

    with open('datasets/user_data.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    try:
        index_to_update = [row[0] for row in rows].index(current_name)

        rows[index_to_update][0] = new_name

        with open('datasets/user_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    except ValueError:
        print("Name not found in the CSV file.")
