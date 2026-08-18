import json
import operator
import sqlite3
from typing import Annotated, List, Literal, Optional, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from logger import TrajectoryLogger
from plan import Gener, Plan
from tools import check_answer, generate_question

from kb import search_info
from llm import llm


class PlanState(TypedDict):
    student_id: str
    topic_id: str
    input_text: str
    task_type: str
    status: str
    messages: Annotated[List, operator.add]
    goal: str
    plan: List[str]  # список кроків (план підготовки)
    results: List[dict]  # результати виконаних кроків
    completed: int  # відсоток виконаного плану
    topic: str
    grade: int
    past_steps: Annotated[
        List[str], operator.add
    ]  # Історія виконаних дій та результатів
    current_step_idx: int  # Індекс поточного кроку
    tool_call_count: int  # Кількість викликів інструменту


# ==========================================
# ЗАХИСНІ КОНФІГУРАЦІЇ
# ==========================================
MAX_ITERATIONS = 2  # Максимальна кількість викликів інструменту
MAX_EVAL_STEPS = 5  # Захист max_steps для внутрішнього циклу Evaluator


available_tools = [search_info]
# tool_node = ToolNode(available_tools)

llm_with_tools = llm.bind_tools(available_tools, tool_choice="any")
llm_planner = llm.with_structured_output(Plan)

risk_tools = []


# tools_by_name = {t.name: t for t in available_tools}


def dispatch_node(
    state: PlanState,
) -> Command[Literal["planner", "generator", "evaluator", "helper"]]:
    """Розподіляє запити між вузлами."""

    task_type = state.get("task_type", "")
    match task_type:
        case "plan":
            return Command(goto="planner")
        case "gener":
            return Command(goto="generator", update={"task_type": "gener"})
        case "eval":
            return Command(goto="evaluator", update={"task_type": "eval"})
        case _:
            return Command(goto="helper", update={"task_type": "help"})


def plan_node(
    state: PlanState,
) -> Command[Literal["generator", "evaluator", "helper"]]:
    """Розробляє план підготовки студента."""

    topic_id = state.get("topic_id", "невідома тема")
    topic = state.get("topic", "невідома тема")

    kb_results = knowledge_base.query(query_texts=[topic], n_results=3)
    context = ""
    if kb_results["documents"]:
        docs = kb_results["documents"][0]
        context = f"КОНТЕКСТ:\n{'\n---\n'.join(docs)}"

    prompt = [
        SystemMessage(
            content=(
                "Ти — методист-планувальник занять з математики.\n"
                "Склади послідовний план з 5 кроків (ОБОВ'ЯЗКОВО 5 КРОКІВ) "
                f"для підготовки студента до іспиту з математики теми '{topic}'.\n"
                "План має забезпечити послідовне переходу від базових визначень до складних застосувань\n\n"
                "Наприклад. Тема 'Математичний аналіз. Границі'\n"
                "Крок 1. Точка прямування. Односторонні границі. Невизначеності\n"
                "Крок 2. Невизначеності\n"
                "Крок 3. Методи обчислення границь\n"
                "Крок 4. Чудові границі\n"
                "Крок 5. Правило Лопіталя\n\n"
                f"{context}"
            )
        ),
        HumanMessage(content=f"Тема: '{topic_id}'"),
    ]

    plan_obj: Plan = cast(Plan, llm_planner.invoke(prompt))

    # print(f"planner_node - plan: {plan_obj}")

    # print(f"planner_node - return: {result}")

    return Command(
        goto="generator",
        update={
            "task_type": "gener",
            "goal": plan_obj.goal,
            "plan": plan_obj.steps,
            "current_step_idx": 0,
            "past_steps": [f"Складено план з {len(plan_obj.steps)} кроків."],
        },
    )


def gener_node(
    state: PlanState,
) -> Command[Literal["planner", "generator", "evaluator", "helper"]]:
    """Генерує питання на задану тему."""
    # topic_id = state.get("topic_id", "невідома тема")
    topic = state.get("topic", "невідома тема")
    current_step_idx = state.get("current_step_idx", 0)
    completed = state.get("completed", 0)
    results = state.get("results", [])

    plan = state.get("plan", [])
    total_steps = len(plan)

    if total_steps == 0:
        print("План порожній")
        return Command(goto="planner", update={"task_type": "plan"})

    if completed == total_steps:
        print("План виконано")
        with open("questions.json", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n\n"
            )
            print("Список питань збережено в файлі 'questions.json'")
        return Command(goto="evaluator", update={"task_type": "eval"})

    current_step = plan[current_step_idx]

    gener: Gener = cast(
        Gener,
        generate_question.invoke(
            {
                "topic": f"{topic}\n{current_step}",
                "difficulty": "середня",
            }
        ),
    )
    results.append(
        {"topic": gener.topic, "question": gener.question, "answer": gener.answer}
    )
    current_step_idx += 1
    completed = len(results)
    return Command(
        goto="generator",
        update={
            "task_type": "gener",
            "results": results,
            "current_step_idx": current_step_idx,
            "completed": completed,
        },
    )


def eval_node(
    state: PlanState,
) -> Command[Literal["evaluator", "helper", END]]:
    """Перевіряє відповідь студента."""

    results = state.get("results", [])

    for item in results:
        if item.get("checked", False):
            print(f"Питання перевірене:\n'{item['question']}'\n")
            continue

        print(f"Перевіряємо питання '{item['question']}'")
        check = check_answer.invoke(
            {
                "question": item.get("question", "невідоме питання"),
                "student_answer": item.get("answer", "невідома відповідь"),
                "correct_answer": item.get("answer", "невідома відповідь"),
            }
        )
        item["checked"] = True
        item["rate"] = str(check)

        with open("questions.json", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n\n"
            )
            print("Список питань збережено в файлі 'questions.json'")

        return Command(
            goto="evaluator",
            update={
                "task_type": "eval",
                "results": results,
            },
        )
    else:
        print("Всі питання перевірено")

        return Command(goto="helper", update={"task_type": "help"})


def submit_node(
    state: PlanState,
) -> Command[Literal["submitter", "helper", END]]:
    """Підтверджує оцінки виставлені агентом."""

    # Тут має бути логіка взаємодії з викладачем та LMS
    #

    return Command(goto="submitter")


def help_node(
    state: PlanState,
) -> Command[Literal["tools", END]]:
    """Робить пояснення концепцій."""

    tool_call_count = state.get("tool_call_count", 0)

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

    response = llm_with_tools.invoke(prompt)

    # Визначаємо наступний крок
    if response.tool_calls:
        return Command(
            goto="tools",
            update={"messages": [response], "tool_call_count": tool_call_count + 1},
        )
    else:
        return Command(goto=END)


conn = sqlite3.connect("agent_state.db", check_same_thread=False)
saver = SqliteSaver(conn)

plan_workflow = StateGraph(PlanState)

plan_workflow.add_node("dispatcher", dispatch_node)
plan_workflow.add_node("planner", plan_node)
plan_workflow.add_node("generator", gener_node)
plan_workflow.add_node("evaluator", eval_node)
plan_workflow.add_node("helper", help_node)
plan_workflow.add_node("submitter", submit_node)


plan_workflow.add_edge(START, "dispatcher")

tool_node = ToolNode([search_info])
plan_workflow.add_node("tools", tool_node)
plan_workflow.add_edge("tools", "helper")

logger_callback = TrajectoryLogger()

app = plan_workflow.compile(
    checkpointer=saver,
    interrupt_after=["planner", "generator", "evaluator"],
    interrupt_before=["submitter"],
)
