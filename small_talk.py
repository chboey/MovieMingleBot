import re
from datetime import datetime, timedelta
import python_weather
import asyncio
import os


async def getweather(user_input):
    async with python_weather.Client() as client:
        weather = await client.get('United Kingdom')

        if re.search(r'\b(?:today|now)\b', user_input, re.IGNORECASE):
            print(f"Today's temperature is: {weather.current.temperature}°C")
            print(f"Description: {weather.current.description}")
            print(f"Humidity: {weather.current.humidity}%")
            print(f"Precipitation: {weather.current.precipitation} mm")
            print(f"Wind Speed: {weather.current.wind_speed} km/h")
            print(f"Wind Direction: {weather.current.wind_direction}")

        elif re.search(r'\b(?:tomorrow|next day)\b', user_input, re.IGNORECASE):
            print("Bot: Here are the weather details for tomorrow")

            for forecast in weather.forecasts:
                print(f"Date: {forecast.date.today() + timedelta(days=1)}")
                print(f"Moon Phase: {forecast.astronomy.moon_phase}")
                print(f"Sun Rise: {forecast.astronomy.sun_rise}")
                print(f"Sun Set: {forecast.astronomy.sun_set}")
                print(f"Temperature: {forecast.temperature}°C")
                print(f"Lowest Temperature: {forecast.lowest_temperature}°C")
                print(f"Highest Temperature: {forecast.highest_temperature}°C")
                print(f"Sunlight: {forecast.sunlight} hours")
                break
        else:
            print("Bot: Here are the weather details for the next few days")

            for forecast in weather.forecasts:
                print(f"Date: {forecast.date}")
                print(f"Moon Phase: {forecast.astronomy.moon_phase}")
                print(f"Sun Rise: {forecast.astronomy.sun_rise}")
                print(f"Sun Set: {forecast.astronomy.sun_set}")
                print(f"Temperature: {forecast.temperature}°C")
                print(f"Lowest Temperature: {forecast.lowest_temperature}°C")
                print(f"Highest Temperature: {forecast.highest_temperature}°C")
                print(f"Sunlight: {forecast.sunlight} hours")
                print()


def bot_enquiries(intent, user_input):
    if intent == "bot_enquiries_time":
        current_time = datetime.now().time()
        print(f"Bot: The current time is...{current_time}")

    elif intent == "bot_enquiries_weather":
        if os.name == 'nt':
            print(f"Bot: Give me a moment, I'm grabbing weather details!")
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        asyncio.run(getweather(user_input))

    elif intent == "bot_enquiries_sentimental":
        print("Bot: I am bot with no built-in functions for feelings, but I would be happy to "
              "help you with anything you need :D")
