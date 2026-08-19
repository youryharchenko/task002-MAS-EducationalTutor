import sys

from huey.consumer import Consumer

from signals import logger

# 1. Імпортуємо наш екземпляр huey з файлу tasks.py
from tasks import huey


def run_consumer():
    """Кастомний запускник воркера Huey."""
    print("🚀 Запуск Huey Worker Consumer...")
    logger.info("🚀 Запуск Huey Worker Consumer...")

    # 2. Створюємо екземпляр Consumer
    # Передаємо йому наш об'єкт huey та налаштування
    consumer = Consumer(
        huey=huey,
        workers=2,  # Кількість паралельних процесів/воркерів
        worker_type="thread",  # Тип воркерів: 'thread', 'process' або 'greenlet'
        check_worker_health=True,  # Перевірка стану воркерів
    )

    # 3. Запускаємо циклічне слухання черги
    try:
        consumer.run()
    except KeyboardInterrupt:
        print("\n🛑 Воркер зупинено користувачем.")
        sys.exit(0)


if __name__ == "__main__":
    run_consumer()
