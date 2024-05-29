# 🎬 MovieMingleBot

MovieMingleBot is an Interactive NLP-based AI system designed to manage cinema bookings and provide a seamless user experience. This bot can handle various queries related to screening times, booking or cancelling tickets, checking the time or weather, and engaging in small talk.

## Introduction
My approach to an Interactive NLP-based AI system revolves around a cinema booking system, called ‘MovieMingleBot’. This system demonstrates methodologies of cosine-similarity and TF-IDF Vectorization for information retrieval, question answering, and intent matching. The chatbot is capable of handling various types of inquiries, enhancing the user experience with clear elements of Conversational Design.

## Installation
To get started with MovieMingleBot, follow these steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/MovieMingleBot.git
    ```

2. **Navigate to the project directory**
    ```bash
    cd MovieMingleBot
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the bot**
    ```bash
    python main.py
    ```

## Usage
Once the bot is running, you can interact with it via the command line. Below are some example interactions:

- **Greeting the bot**
    ```plaintext
    User: Hello!
    Bot: Hi there! How can I assist you today?
    ```

- **Booking a movie ticket**
    ```plaintext
    User: I want to book a movie ticket.
    Bot: Sure! Which movie would you like to watch?
    ```

- **Checking the weather**
    ```plaintext
    User: What's the weather like today?
    Bot: The weather today is sunny with a high of 25°C.
    ```

## Features
- **🎟️ Movie Ticket Booking:** Book or cancel tickets for movies currently screening.
- **🕒 Time Queries:** Check the current time.
- **🌦️ Weather Queries:** Get the weather forecast for today, tomorrow, or the next few days.
- **💬 Small Talk:** Engage in casual conversation with the bot.

## Architecture
The MovieMingleBot architecture includes the following components:

1. **Main Module:** `main.py` - The entry point of the bot.
2. **Intent Recognition:** Uses cosine similarity and TF-IDF vectorization.
3. **Text Preprocessing:** Involves tokenization, part-of-speech tagging, and named entity recognition.
4. **User Interaction:** Handles user registration, login, and session management.
5. **Transaction Management:** Manages movie bookings, seat availability, and cancellations.
6. **Information Retrieval:** Provides details about movies using a predefined dataset.

![Bot Architecture](architecture.png)

## Future Work
- **🤖 Enhanced AI:** Improve the bot's NLP capabilities for more complex queries.
- **📱 Mobile App Integration:** Develop a mobile application for easier access.
- **🌐 Multilingual Support:** Add support for multiple languages to cater to a wider audience.

## Contributing
Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
