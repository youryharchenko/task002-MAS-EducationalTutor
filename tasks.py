from huey import SqliteHuey

huey = SqliteHuey(filename="huey_queue.db")


@huey.task()
def test_task(name: str):
    print(f"Привіт, {name}!")
    return "Тестова задача виконана."
