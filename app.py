from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


# 1. Модель відповіді від LLM
class Weather(BaseModel):
    temperature: float
    condition: str


# 2. Агент Pydantic AI
agent = Agent(
    # В Pydantic AI для OpenAI правильний формат моделі:
    # 'openai:gpt-4o-mini', а не просто 'gpt-4o-mini'
    "openai:gpt-4o-mini",
    output_type=Weather,
    instructions=(
        "You are a weather bot. "
        "Read the user question and respond ONLY with a valid Weather object: "
        "temperature (float, in Celsius) and condition (short English text)."
    ),
)


# 3. FastAPI
app = FastAPI()


class UserQuery(BaseModel):
    question: str


@app.post("/ask")
async def ask_agent(query: UserQuery):
    # запускаємо агента
    result = await agent.run(query.question)

    # структурований результат (твоя модель Weather)
    weather: Weather = result.output

    # usage з токенами
    usage = result.usage()  # RunUsage dataclass

    # Повертаємо чистий JSON, без ```json ... ```
    return {
        "weather": {
            "temperature": weather.temperature,
            "condition": weather.condition,
        },
        "usage": {
            "requests": usage.requests,
            "tool_calls": usage.tool_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "details": usage.details,  # тут ще детальніший розклад, якщо модель його дає
        },
    }
