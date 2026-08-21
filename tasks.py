from huey import SqliteHuey
from langgraph.graph.state import RunnableConfig
from rich.console import Console

from query_agent import QueryAgent, QueryState
from tutor_agent import TutorAgent, TutorState

huey = SqliteHuey(filename="huey_queue.db")


@huey.task()
def test_task(name: str):
    print(f"Привіт, {name}!")
    return "Тестова задача виконана."


@huey.task()
def query_task(student_id: str, topic_id: str, topic: str, query: str):
    thread_id = f"{student_id}/{topic_id}"
    query_input: QueryState = {
        "student_id": student_id,
        "topic_id": topic_id,
        "task_type": "help",
        "topic": topic,
        "input_text": "",
        "messages": [query],
        "status": "",
        "tool_call_count": 0,
    }
    agent = QueryAgent("helper", thread_id)
    try:
        # final_output = app.invoke(query_input, config=config)
        final_output = agent.run(query_input)
        try:
            reply = final_output["messages"][-1].content
        except Exception:
            reply = str(final_output["messages"][-1])
    except Exception as ex:
        reply = str(ex)

    print(f"\n{reply}\n")
    return reply


@huey.task()
def gener_task(student_id: str, topic_id: str, topic: str, query: str):
    thread_id = f"{student_id}/{topic_id}"
    query_input: TutorState = {
        "student_id": student_id,
        "topic_id": topic_id,
        "task_type": "tutor",
        "topic": topic,
        "input_text": "",
        "messages": [query],
        "status": "",
        "tool_call_count": 0,
        "step_count": 0,
        "completed": False,
        "question": "",
        "answer": "",
        "grade": "",
    }
    agent = TutorAgent("tutor", thread_id)
    try:
        # final_output = app.invoke(query_input, config=config)
        final_output = agent.run(query_input)
        try:
            reply = final_output["question"]
        except Exception:
            reply = str(final_output)
    except Exception as ex:
        reply = str(ex)

    print(f"\n{reply}\n")
    return reply
