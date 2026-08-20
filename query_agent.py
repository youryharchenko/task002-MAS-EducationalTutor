import operator
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from base_agent import BaseGraphAgent
from kb import search_info
from llm import llm
from logger import TrajectoryLogger

MAX_ITERATIONS = 2  # Максимальна кількість викликів інструменту


class QueryState(TypedDict):
    student_id: str
    topic_id: str
    input_text: str
    task_type: str
    status: str
    messages: Annotated[list, operator.add]
    topic: str
    tool_call_count: int  # Кількість викликів інструменту


class QueryAgent(BaseGraphAgent):
    def __init__(self, name: str):
        self.llm = llm
        self.tools = [search_info]
        self.logger_callback = TrajectoryLogger(f"{name}_trajectory.json")
        self.tool_node = ToolNode(self.tools)

        super().__init__(name=name)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(QueryState)

        workflow.add_edge(START, "helper")

        workflow.add_node("helper", self._help_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge("tools", "helper")

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)

        return workflow.compile(checkpointer=saver)

    def _help_node(
        self,
        state: QueryState,
    ) -> Command[Literal["tools", END]]:
        """Робить пояснення концепцій."""

        tool_call_count = state.get("tool_call_count", 0)
        last_message = state.get("messages")[-1]

        if (
            tool_call_count > 0
            and last_message.content
            and last_message.status == "success"
        ):
            return Command(goto=END, update={"tool_call_count": 0})

        if tool_call_count > MAX_ITERATIONS:
            print(f"Кількість викликів досягла межі {tool_call_count}")
            return Command(goto=END, update={"tool_call_count": 0})

        messages = state.get("messages", [])

        if not messages:
            print("Нема запитів до консультанта")
            return Command(goto=END)

        prompt = [
            SystemMessage(
                content=(
                    "Ти — досвідчений викладач вищої математики.\n\n"
                    "ОБОВ'ЯЗКОВО. Користуйся інструментом 'search_info'"
                    # f"{context}\n\n"
                    f"Твоє завдання: Пояснити поняття, які наведені в питанні."
                    f"Питання: '{messages[0]}'.\n\n"
                )
            ),
            HumanMessage(content=f"Дай пояснення на питаня: {messages[0]}"),
        ]

        response = self.llm.bind_tools(self.tools).invoke(prompt)

        # Визначаємо наступний крок
        if response.tool_calls:
            return Command(
                goto="tools",
                update={"messages": [response], "tool_call_count": tool_call_count + 1},
            )
        else:
            return Command(goto=END)
