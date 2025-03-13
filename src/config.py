from os import getenv
from dotenv import find_dotenv, load_dotenv


def load():
    if not getenv('PROD'):
        load_dotenv(find_dotenv())
