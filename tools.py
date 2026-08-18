import re
from typing import Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kb import new_kb, search_info
from llm import llm
from plan import Gener

knowledge_base = new_kb()


class GenerateQuestionInput(BaseModel):
    """Схема валідації вхідних даних для генерації тестових питань."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    topic: str = Field(
        ..., min_length=3, max_length=150, description="Тема тестового питання"
    )
    difficulty: Literal["низька", "середня", "висока"] = Field(
        "середня", description="Рівень складності тестового питання"
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        # 1. Захист від порожніх рядків після стрипингу
        if not value:
            raise ValueError("Тема питання не може бути порожньою.")

        # 2. Перевірка на символи (control characters / newlines), які можуть ламати промпт
        # if re.search(r"[\r\n\t\x00-\x1f]", value):
        #    raise ValueError(
        #        "Тема не повинна містити переносів рядків або спецсимволів."
        #    )

        return value


@tool("generate_question", args_schema=GenerateQuestionInput)
def generate_question(
    topic: str, difficulty: Literal["низька", "середня", "висока"]
) -> Gener:
    """
    Генерує тестове питання на задану тему вказаної складності.

    Args:
        topic str "Тема тестового питання"
        difficulty ("низька", "середня", "висока") "Рівень складності тестового питання"
        )

    Returns:
         str "Тестове питання"
    """

    kb_results = knowledge_base.query(query_texts=[topic], n_results=3)
    context = ""
    if kb_results["documents"]:
        docs = kb_results["documents"][0]
        context = f"КОНТЕКСТ:\n{'\n---\n'.join(docs)}"

    prompt = [
        SystemMessage(
            content=(
                "Ти — досвідчений викладач вищої математики.\n\n"
                # "Користуйся інструментом 'search_info'"
                f"{context}\n\n"
                f"Твоє завдання: згенерувати 1 якісне контрольне питання та детальну еталонну відповідь "
                f"до теми: '{topic}', складність: '{difficulty}'.\n\n"
                "Вимоги:\n"
                "1. 'question' має бути конкретним математичним питанням чи задачею (НЕ заголовком).\n"
                "2. 'answer' МУСИТЬ містити повне розв'язання чи математичне пояснення (не залишай порожнім!).\n\n"
                "ПРИКЛАД ЯКІСНОГО ВИВОДУ:\n"
                "Topic: Вектори в R^n\n"
                "Question: Дано вектори u = (1, 2) та v = (-3, 1). Знайдіть результат лінійної комбінації 2u - v та поясніть її геометричний зміст.\n"
                "Answer: 2u - v = 2*(1, 2) - (-3, 1) = (2, 4) + (3, -1) = (5, 3). Геометрично це відповідає додаванню вектора 2u та протилежного вектора до v за правилом паралелограма."
            )
        ),
        HumanMessage(content=f"Склади питаня до теми: '{topic}'"),
    ]

    llm_generator = llm.with_structured_output(Gener)

    gener: Gener = cast(Gener, llm_generator.invoke(prompt))

    return gener


class CheckAnswerInput(BaseModel):
    """
    Схема валідації вхідних даних для перевірки відповіді.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    question: str = Field(
        ..., min_length=5, max_length=2000, description="Тестове питання"
    )
    student_answer: str = Field(
        ..., min_length=5, max_length=5000, description="Відповідь студента на питання"
    )
    correct_answer: str = Field(
        ..., min_length=5, max_length=5000, description="Правильна відповідь на питання"
    )

    @field_validator("student_answer")
    @classmethod
    def validate_student_answer(cls, value: str) -> str:
        """Валідація та санітизація відповіді студента."""
        cleaned_value = value.strip()

        # 1. Обробка відсутності відповіді або коротких маркерів
        if not cleaned_value or cleaned_value in {"-", "не знаю", "no answer"}:
            return "Студент не надав відповіді."


@tool("check_answer", args_schema=CheckAnswerInput)
def check_answer(question: str, student_answer: str, correct_answer: str) -> str:
    """
    Перевіряє відповідь студента.

    Args:
        question str = "Тестове питання"
        student_answer str "Відповідь студента на питання"
        correct_answer str "Правильна відповідь на питання"

    Returns:
         str "Оцінка відповіді студента"
    """

    kb_results = knowledge_base.query(query_texts=[question], n_results=3)
    context = ""
    if kb_results["documents"]:
        docs = kb_results["documents"][0]
        context = f"КОНТЕКСТ:\n{'\n---\n'.join(docs)}"

    prompt = [
        SystemMessage(
            content=(
                "Ти — досвідчений викладач вищої математики.\n\n"
                # "Користуйся інструментом 'search_info'"
                f"{context}\n\n"
                f"Твоє завдання: перевірити якість відповіді студента на питання "
                f"Питання: '{question}'.\n\n"
                f"Відповідь студента: '{student_answer}'.\n\n"
                "Вимоги:\n"
                "Оцінка має бути з переліку: 'незадовільно', 'задовільно', 'добре', 'відмінно'\n"
            )
        ),
        HumanMessage(
            content=f"Дай оцінку відповіді на питаня: 'питання: {question}, відповідь студента: '{student_answer}'"
        ),
    ]

    # llm_evaluator = llm.bind_tools([search_info])
    # ai_mess = llm_evaluator.invoke(prompt)
    ai_mess = llm.invoke(prompt)
    return str(ai_mess.content)


class ExplainConceptInput(BaseModel):
    """
    Схема валідації вхідних даних для пояснення концепції.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    concept: str = Field(..., description="Концепція")
    level: Literal["коротко", "детально"] = Field(
        ..., description="Рівень пояснення концепції"
    )


@tool("explain_concept", args_schema=ExplainConceptInput)
def explain_concept(concept: str, level: Literal["коротко", "детально"]) -> str:
    """
    Дає коротке або детальне пояснення концепції.

    Args:
         concept str "Концепція"
         level str ("коротко", "детально") "Рівень пояснення концепції"

    Returns:
         str "Пояснення концепції."
    """

    return "Це має бути пояснення концепції."


class SubmitGradeInput(BaseModel):
    """
    Схема валідації вхідних даних для виставлення оцінки в LMS.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    student_id: str = Field(..., description="Ідентифікатор студента")
    assignment: str = Field(..., description="Ідентифікатор практичної роботи")
    grade: Literal["незадовільно", "задовільно", "добре", "відмінно"] = Field(
        ..., description="Оцінка практичної роботи"
    )


@tool("submit_grade", args_schema=SubmitGradeInput)
def submit_grade(
    student_id: str,
    assignment: str,
    grade: Literal["незадовільно", "задовільно", "добре", "відмінно"],
) -> str:
    """
    Виставляє оцінку в LMS.

    Args:
        student_id str "Ідентифікатор студента"
        assignment str "Ідентифікатор практичної роботи"
        grade str ("незадовільно", "задовільно", "добре", "відмінно"] "Оцінка практичної роботи"

    Returns:
        str "Результат виставлення оцінки."
    """

    return "задовільно"
