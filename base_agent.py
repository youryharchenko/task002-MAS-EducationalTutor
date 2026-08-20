from abc import ABC, abstractmethod
from typing import Any, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot


class BaseGraphAgent(ABC):
    """Базовий абстрактний клас для всіх LangGraph агентів навчальної системи."""

    def __init__(self, name: str):
        self.name = name
        self.db_path = f"{name}_state.db"
        # Кожен дочірній клас повинен зібрати свій граф
        self.app: CompiledStateGraph = self._compile_graph()

    @abstractmethod
    def _build_graph(self) -> Any:
        """Абстрактний метод: побудова топології StateGraph (вузли та ребра)."""
        pass

    def _compile_graph(self) -> CompiledStateGraph:
        """Внутрішня компіляція графа з підключенням чекпоїнтера."""
        workflow = self._build_graph()
        # Повертаємо скомпільований граф. Чекпоїнтер передаватиметься при виклику invoke/stream,
        # або підключається тут контекстно.
        return workflow

    # --- УНІФІКОВАНИЙ ПУБЛІЧНИЙ ІНТЕРФЕЙС ---

    def run(self, inputs, config) -> dict[str, Any]:
        """Запуск або початок нової сесії агента."""
        return self.app.invoke(inputs, config=config)

    def resume(self, resume_data, config) -> dict[str, Any]:
        """Відновлення виконання агента після HITL (interrupt)."""
        return self.app.invoke(Command(resume=resume_data), config=config)

    def get_state(self, config) -> StateSnapshot:
        """Отримання поточного стану та снапшоту сесії."""
        return app.get_state(config)
