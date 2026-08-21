import json
import pathlib
import shlex
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console

from kb import chroma_client, new_kb
from tasks import gener_task, huey, query_task, test_task

# Команди та їхні ариті
commands = {
    "new-kb": 1,
    "add-kb": 1,
    "del-kb": 1,
    "ls-kb": 0,
    "exit": 0,
    "quit": 0,
    "close": 0,
    "hello": 0,
    "help": 0,
    "test-task": 1,
    "status-task": 1,
    "gener-task": 1,
}

STUDENT_ID = "Студент01"
TOPIC_ID = "Вектори"
TOPIC = "Лінійна алгебра. Вектори."


def main():
    """Вхідна точка для освітнього репетитора"""
    script_path = pathlib.Path(__file__)
    # dict_path = script_path.with_name("contacts.pickle")
    # dictionary = read_dict(dict_path)

    history_path = script_path.with_name(".history")
    history = FileHistory(history_path)
    completer = WordCompleter(list(commands.keys()))

    console = Console()

    console.print("[bold green]Вас вітає помічник з навчання математиці![/bold green]")
    console.print(
        "Type 'help' for available commands or 'exit' | 'quit' | 'close' to quit."
    )
    console.print(
        "Натискайте [yellow]Tab[/yellow] для автоматичного доповнення команди."
    )
    session = PromptSession(
        history=history, completer=completer, reserve_space_for_menu=True
    )

    def print_help():
        console.print("Доступні команди та кількість параметрів:")
        for k, v in commands.items():
            console.print(f"    {k}/{v}")

    def parse_input(msg_prompt: str) -> list[str]:
        msg = session.prompt(msg_prompt)
        res = shlex.split(msg)
        if len(res) > 0:
            res[0] = res[0].strip().lower()
            arity = len(res) - 1
            if res[0] in commands:
                if commands[res[0]] == arity:
                    for i in range(arity):
                        res[i + 1] = res[i + 1].strip()
                else:
                    console.print(
                        f"[bold red]Команда '{res[0]}' очікує {commands[res[0]]}, але отримала {arity} параметр(ів)[/bold red]"
                    )
                    res = ["error"]
        else:
            res = ["error"]

        return res

    knowledge_base = new_kb()

    while True:
        repl = parse_input("Enter a command: ")
        try:
            match repl:
                case ["new-kb", name]:
                    knowledge_base = new_kb(name)

                case ["add-kb", name]:
                    with open(name, "r") as f:
                        documents = json.load(f)
                        doc_ids = [f"doc_{i}" for i in range(len(documents))]
                        knowledge_base.add(documents=documents, ids=doc_ids)

                case ["ls-kb"]:
                    kb_ls = chroma_client.list_collections()
                    if kb_ls:
                        console.print(
                            "\n",
                            "\n".join(f"    {kb.name}({kb.count()})" for kb in kb_ls),
                            "\n",
                        )
                    else:
                        console.print(
                            "[bold yellow]Нема жодної бази знань![/bold yellow]"
                        )
                case ["exit"] | ["quit"] | ["close"]:
                    console.print("[bold green]До побачення![/bold green]")
                    break
                case ["hello"]:
                    console.print("[bold green]Чим можу вам допомогти?[/bold green]")
                case ["help"]:
                    print_help()
                case ["status-task", task_id]:
                    # Отримуємо обгортку результату з брокера за ID
                    result = huey.result(task_id, preserve=True)
                    if result is None:
                        print("Статус: ⏳ В черзі (Pending) або не знайдено")
                    else:
                        print("Статус: ✅ Виконано успішно")
                        print(f"Результат: {result}")
                case ["test-task", name]:
                    result_wrapper = test_task(name)
                    console.print(
                        f"[bold green]Task ID: {result_wrapper.id}[/bold green]"
                    )
                case ["error"]:
                    console.print("")
                case ["gener-task", question]:
                    result_wrapper = gener_task(STUDENT_ID, TOPIC_ID, TOPIC, question)
                    console.print(
                        f"[bold green]Task ID: {result_wrapper.id}[/bold green]"
                    )
                case _:
                    query = " ".join(repl)
                    console.print(
                        f"Запит: [bold green]{query}[/bold green]\n",
                        "Передаємо питання LLM",
                    )
                    # start_time = time.time()
                    result_wrapper = query_task(STUDENT_ID, TOPIC_ID, TOPIC, query)
                    console.print(
                        f"[bold green]Task ID: {result_wrapper.id}[/bold green]"
                    )
                    # elapsed_time = round(time.time() - start_time, 2)
                    # console.print(f"{elapsed_time} сек.")

        except Exception as ex:
            console.print(f"[bold red]{ex}[/bold red]")


if __name__ == "__main__":
    main()
