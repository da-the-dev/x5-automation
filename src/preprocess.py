import pandas as pd
import re
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def clear_spaces_inside(text):
    words = text.split()
    words = list(map(lambda x: x.strip(), words))
    text_clear = " ".join(words)

    return text_clear


def preprocess(text):

    greeting_words = {
        "здравствуйте",
        "здравствуй",
        "привет",
        "приветствую",
        "добрый",
        "день",
        "утро",
        "вечер",
        "ночь",
    }

    polite_words = {
        "пожалуйста",
        "пож",
        "будь",
        "добрый",
        "спасибо",
        "благодарю",
        "прошу",
        "спс",
        "плиз",
        "плз",
    }

    self_intro_words = {"я", "меня", "зовут", "будучи", "являюсь"}

    quest_words = {"как", "где", "какой"}

    request_verbs = {
        "хотеть",
        "просить",
        "помогать",
        "надо",
        "нужно",
        "требовать",
        "просьба",
        "возможность",
        "необходимо",
        "мочь",
    }
    roles = {
        "hr",
        "дм",
        "менеджер",
        "руководитель",
        "специалист",
        "сотрудник",
        "работник",
        "директор",
        "начальник",
        "администратор",
    }
    profanity = {
        "блять",
        "бля",
        "сука",
        "пиздец",
        "хуй",
        "нахуй",
        "хрен",
        "нахрен",
        "хуйня",
        "пизда",
        "ебать",
        "ебанина",
        "заебал",
        "заебало",
    }

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    link_pattern = r"(https?://[^\s]+|www\.[^\s]+)"

    text = text.lower().strip()
    text = re.sub(email_pattern, "MAIL", text)
    text = re.sub(link_pattern, "LINK", text)
    text = re.sub("\+7 \(xxx\) xxx xx xx", "PHONE", text)
    text = clear_spaces_inside(text)

    glossary_dict = {
        "лк": "личный кабинет",
        "бир": "беременность и роды",
        "зп": "заработная плата",
        "ндфл": "налог на доходы физических лиц",
        "стд": "срочный трудовой договор",
        "тк": "трудовой договор",
        "ао": "авансовый отчет",
        "sla": "сроки",
        "эцп": "электронная цифровая подпись",
        "кр": "кадровый резерв",
    }

    for key, value in glossary_dict.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)

    tokens = []
    for sent in re.split(r"[.,!?;:()\[\]{}/-]", text.lower()):
        tokens.extend(sent.split())

    tokens_to_remove = []
    for token in tokens:
        normalized_word = morph.parse(token)[0].normal_form
        if (
            normalized_word in greeting_words
            or normalized_word in polite_words
            or normalized_word in self_intro_words
            or normalized_word in quest_words
            or normalized_word in request_verbs
            or normalized_word in profanity
            or normalized_word in roles
        ):
            tokens_to_remove.append(token)
    filtered_tokens = [token for token in tokens if token not in tokens_to_remove]
    filtered_text = " ".join(filtered_tokens)
    filtered_text = re.sub(r"\s+", " ", filtered_text)
    return filtered_text.strip()


def preprocess(text: str) -> str:
    text = clear_spaces_inside(text)
    text = filter_irrelevant_info(text)

    return text
