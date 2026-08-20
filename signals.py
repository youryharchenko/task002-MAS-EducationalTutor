import logging

from huey import SqliteHuey
from huey.signals import (
    SIGNAL_COMPLETE,
    SIGNAL_ERROR,
    SIGNAL_EXECUTING,
    SIGNAL_RETRYING,
)

from tasks import huey

# Налаштовуємо логування
logging.basicConfig(
    filename="app_execution.log",  # Назва файлу логів
    filemode="w",  # 'a' — дозапис (append), 'w' — перезапис при кожному старті
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("HueySignals")

# huey = SqliteHuey(filename="huey_queue.db")


# 1. Сигнал: Початок виконання задачі
@huey.signal(SIGNAL_EXECUTING)
def on_task_executing(signal, task):
    thread_id = task.args[0] if task.args else "Unknown"
    logger.info(
        f"🚀 [SIGNAL] Почато виконання задачі '{task.name}' (Task ID: {task.id}, Thread: {thread_id})"
    )


# 2. Сигнал: Успішне завершення
@huey.signal(SIGNAL_COMPLETE)
def on_task_completed(signal, task, retval=None):
    logger.info(
        f"✅ [SIGNAL] Задача '{task.name}' успішно виконана! (Task ID: {task.id})"
    )


# 3. Сигнал: Помилка виконання (наприклад, збій LLM)
@huey.signal(SIGNAL_ERROR)
def on_task_error(signal, task, exc=None):
    logger.error(
        f"❌ [SIGNAL] Збій у задачі '{task.name}' (Task ID: {task.id})! Помилка: {exc}"
    )


# 4. Сигнал: Повторна спроба (Retry)
@huey.signal(SIGNAL_RETRYING)
def on_task_retrying(signal, task, exc=None):
    logger.warning(
        f"🔄 [SIGNAL] Перезапуск задачі '{task.name}' (Task ID: {task.id}) через помилку: {exc}"
    )
