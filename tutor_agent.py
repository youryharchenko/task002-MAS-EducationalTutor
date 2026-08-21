import json
import operator
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from base_agent import BaseGraphAgent
from kb import search_info
from llm import llm
from logger import TrajectoryLogger

MAX_ITERATIONS = 2  # Максимальна кількість викликів інструменту
MAX_STEPS = 3


class TutorState(TypedDict):
    student_id: str
    topic_id: str
    input_text: str
    task_type: str
    status: str
    messages: Annotated[list, operator.add]
    completed: bool
    topic: str
    grade: str
    question: str
    answer: str
    tool_call_count: int  # Кількість викликів інструменту
    step_count: int  # Кількість циклів генрації


class TutorAgent(BaseGraphAgent):
    def __init__(self, name: str, thread_id: str):
        self.llm = llm
        self.tools = [search_info]
        self.logger_callback = TrajectoryLogger(f"{name}_trajectory.json")
        self.tool_node = ToolNode(self.tools)
        config = {
            "recursion_limit": 15,
            "callbacks": [self.logger_callback],
            "configurable": {"thread_id": thread_id},
        }
        super().__init__(name=name, config=config)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(TutorState)

        workflow.add_edge(START, "generator")

        workflow.add_node("generator", self._gener_node)
        workflow.add_node("evaluator", self._eval_node)

        workflow.add_node("tools", self.tool_node)
        workflow.add_edge("tools", "generator")

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)

        return workflow.compile(checkpointer=saver, interrupt_before=["evaluator"])

    def _gener_node(
        self,
        state: TutorState,
    ) -> Command[Literal["evaluator", END]]:
        """Генерує питання на задану тему."""
        # topic_id = state.get("topic_id", "невідома тема")
        topic = state.get("topic", "невідома тема")
        completed = state.get("completed", 0)

        if completed:
            print("Завдання вже виконано")
            return Command(goto=END, update={"task_type": "end"})

        tool_call_count = state.get("tool_call_count", 0)
        step_count = state.get("step_count", 0)
        last_message = state.get("messages")[-1]

        if (
            tool_call_count > 0
            and last_message.content
            and last_message.status == "success"
        ):
            return Command(
                goto="evaluator", update={"tool_call_count": 0, "task_type": "eval"}
            )

        if step_count > MAX_STEPS:
            print(f"Кількість циклів генерації {step_count} досягла межі")
            return Command(goto=END, update={"step_count": 0})

        if tool_call_count > MAX_ITERATIONS:
            print(f"Кількість викликів досягла межі {tool_call_count}")
            return Command(goto="evaluator", update={"tool_call_count": 0})

        messages = state.get("messages", [])
        if not messages:
            messages = [topic]

        prompt = [
            SystemMessage(
                content=(
                    "Ти — досвідчений викладач вищої математики.\n\n"
                    # "Користуйся інструментом 'search_info'"
                    # f"{context}\n\n"
                    f"Твоє завдання: згенерувати 1 (одне) якісне контрольне питання та детальну еталонну відповідь "
                    f"до теми: '{topic}''.\n\n"
                    "Вимоги до формату:\n"
                    "Це має бути JSON без зайвих префіксів та суфіксів (типу ```json).\n"
                    "Поле 'question' має бути конкретним математичним питанням чи задачею (НЕ заголовком). Без вступу і міркувань.\n"
                    "Поле 'answer' МУСИТЬ містити розв'язання чи коротке математичне пояснення (не залишай порожнім!).\n\n"
                    "ПРИКЛАД ЯКІСНОГО ВИВОДУ:\n"
                    "question: Дано вектори u = (1, 2) та v = (-3, 1). Знайдіть результат лінійної комбінації 2u - v.\n"
                    "answer: 2u - v = 2*(1, 2) - (-3, 1) = (2, 4) + (3, -1) = (5, 3)."
                )
            ),
            HumanMessage(content=f"Склади питаня до теми: '{topic}'"),
        ]

        response = self.llm.bind_tools(self.tools).invoke(prompt)

        # Визначаємо наступний крок
        if response.tool_calls:
            return Command(
                goto="tools",
                update={"messages": [response], "tool_call_count": tool_call_count + 1},
            )
        else:
            reply = response.content
            try:
                reply_json = json.loads(reply)
            except Exception as ex:
                return Command(
                    goto="generator",
                    update={
                        "tool_call_count": 0,
                        "step_count": step_count + 1,
                        "messages": [str(ex)],
                    },
                )

            return Command(
                goto="evaluator",
                update={
                    "tool_call_count": 0,
                    "question": reply_json.get("question", "відсутнє"),
                    "answer": reply_json.get("answer", "відсутній"),
                },
            )

    def _eval_node(
        self,
        state: TutorState,
    ) -> Command[Literal["tools", "generator", END]]:
        """Перевіряє відповідь студента."""

        tool_call_count = state.get("tool_call_count", 0)
        step_count = state.get("step_count", 0)
        last_message = state.get("messages")[-1]
        question = state.get("question", "")
        answer = state.get("answer", "")

        if (
            tool_call_count > 0
            and last_message.content
            and last_message.status == "success"
        ):
            return Command(goto=END, update={"tool_call_count": 0})

        if step_count > MAX_STEPS:
            print(f"Кількість циклів генерації {step_count} досягла межі")
            return Command(goto=END, update={"step_count": 0})

        if tool_call_count > MAX_ITERATIONS:
            print(f"Кількість викликів досягла межі {tool_call_count}")
            return Command(
                goto="generator",
                update={"tool_call_count": 0, "step_counr": step_count + 1},
            )

        messages = state.get("messages", [])

        if not messages:
            messages = [question]

        prompt = [
            SystemMessage(
                content=(
                    "Ти — досвідчений викладач вищої математики.\n\n"
                    "Користуйся інструментом 'search_info'"
                    # f"{context}\n\n"
                    f"Твоє завдання: перевірити якість відповіді студента на питання "
                    f"Питання: '{question}'.\n\n"
                    f"Відповідь студента: '{answer}'.\n\n"
                    "Вимоги:\n"
                    "Оцінка має бути з переліку: 'незадовільно', 'задовільно', 'добре', 'відмінно'\n"
                )
            ),
            HumanMessage(
                content=f"Дай оцінку відповіді на питаня: 'питання: {question}, відповідь студента: '{answer}'"
            ),
        ]

        response = self.llm.bind_tools(self.tools).invoke(prompt)

        # Визначаємо наступний крок
        if response.tool_calls:
            return Command(
                goto="tools",
                update={"messages": [response], "tool_call_count": tool_call_count + 1},
            )
        else:
            return Command(goto=END, update={"grade": response.content})
