from pydantic import BaseModel, Field


class Plan(BaseModel):
    """План підготовки до іспиту."""

    goal: str = Field(description="Головна ціль підготовки до іспиту")
    steps: list[str] = Field(
        description="Список кроків для досягнення цілі підготовки до іспиту"
    )


class Gener(BaseModel):
    """Згенероване питання."""

    topic: str = Field(description="Тема питання")
    question: str = Field(
        description="Сформульоване розгорнуте контрольне питання або задача для студента (наприклад: 'Що таке лінійна комбінація векторів та як її обчислити?')"
    )
    answer: str = Field(
        description="Повна, детальна та правильна Etalon-відповідь на це питання"
    )
