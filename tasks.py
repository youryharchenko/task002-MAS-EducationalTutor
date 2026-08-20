from huey import SqliteHuey
from rich.console import Console

from query_agent import QueryAgent

huey = SqliteHuey(filename="huey_queue.db")


@huey.task()
def test_task(name: str):
    print(f"Привіт, {name}!")
    return "Тестова задача виконана."


@huey.task()
def query_task(query_input, config):
    agent = QueryAgent("helper")
    try:
        # final_output = app.invoke(query_input, config=config)
        final_output = agent.run(query_input, config)
        try:
            reply = final_output["messages"][-1].content
        except Exception:
            reply = str(final_output["messages"][-1])
    except Exception as ex:
        reply = str(ex)

    print(f"\n{reply}\n")
    return reply
