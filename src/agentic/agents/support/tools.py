from typing import Optional, Union

from langchain_core.tools import StructuredTool


# Hardcoded list of available classes from RAG metadata
AVAILABLE_RAG_CLASSES = ['автомобиль', 'БиР', 'больничный', 'график работы', 'доверенность',
       'зарплата', 'заявки', 'командировка', 'ЛК', 'материальная помощь',
       'моя карьера', 'МЧД', 'налоговый вычет', 'отгул', 'отпуск',
       'перевод', 'поддержка', 'ЭЦП', 'документооборот',
       'прием на работу', 'СБ', 'справка', 'табель', 'увольнение',
       'удаленная работа', 'уход за больным', 'SED', 'выручай-карта',
       'дмс', 'оператор', 'обучение', 'Отпуск']


async def find_relevant_qa_examples(
        query: str, 
        query_classifications: Optional[list[str]] = None
) -> Union[str, list[str]]:
    """
    A tool for finding relevant QA examples for a given query

    Args:
        query: A query to find relevant examples for
        query_classifications: A list of classifications for the query

    Returns:
        A list of relevant QA examples
    """
    # return Nothing found
    return "Ничего не найдено"


async def call_human_support() -> None:
    """
    A tool for calling human support
    """
    pass


# Creating a structured tool for finding relevant QA examples
find_relevant_qa_examples_tool = StructuredTool.from_function(
    coroutine=find_relevant_qa_examples,
    name="FindRelevantQAExamples",
    description=(
"""
Найти релевантные примеры вопросов и ответов для заданного запроса.

Для повышения качества поиска можно указать тип запрос из списка доступных значений.
"""
    ),
    args_schema={
        "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Запрос для поиска релевантных примеров"
                },
                "query_classifications": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": AVAILABLE_RAG_CLASSES
                    },
                    "description": "Классификации запроса"
                }
            },
            "required": ["query"],
    },
)


# Creating a structured tool for calling human support
сall_human_support_tool = StructuredTool.from_function(
    coroutine=call_human_support,
    name="CallHumanSupport",
    description=(
"""
Позвать человеческую поддержку
"""
    ),
    args_schema={}
)
