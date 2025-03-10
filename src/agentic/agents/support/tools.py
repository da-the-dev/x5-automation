from typing import Optional, Union

from langchain_core.tools import StructuredTool


def find_relevant_qa_examples(query: str) -> Union[str, list[str]]:
    """
    A tool for finding relevant QA examples for a given query

    Args:
        query: A query to find relevant examples for

    Returns:
        A list of relevant QA examples
    """
    # return nothing for now
    # TODO: Implement the actual logic
    return "Ничего не найдено"


def call_human_support() -> None:
    """
    A tool for calling human support
    """
    pass


# Creating a structured tool for finding relevant QA examples
find_relevant_qa_examples_tool = StructuredTool.from_function(
    func=find_relevant_qa_examples,
    name="FindRelevantQAExamples",
    description="Найти релевантные примеры из базы вопросов и ответов (QA) для заданного запроса.",
    args_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Запрос для поиска релевантных ему примеров из базы",
                "examples": [
                    "Как загрузить больничный лист?",
                    "Куда подать новые банковские реквизиты.",
                    "не могу отменить заявку",
                    "Исключить заявку из SLA",
                    "нет заявок",
                    "из раздела 'передать в архив' пропали все документы для передачи МП",
                    "Не работает раздел задачи",
                    "требуется смена номера телефона, у сотрудникка не доступа к старому номеру",
                    "изменить номер телефона в лк сотруднику",
                    "Обновить номер телефона",
                    "Сменить номер телефона сотрудника"
                ]
            },
        },
        "required": ["query"],
    },
)


# Creating a structured tool for calling human support
сall_human_support_tool = StructuredTool.from_function(
    func=call_human_support,
    name="CallHumanSupport",
    description="Пореключить чат на человека.",
    args_schema={}
)
