import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai import Agent
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# --- Модель відповіді ---
class Weather(BaseModel):
    temperature: float
    condition: str
    explication: str
    date: str


app = FastAPI()


class UserQuery(BaseModel):
    question: str


@app.post("/ask")
async def ask_agent(query: UserQuery):
    today_str = date.today().isoformat()

    instructions = (
        f"You are a weather bot. "
        f"Today is {today_str}. "
        f"Return ONLY a valid Weather object with fields: "
        f"temperature (float, Celsius), "
        f"condition (short English text), "
        f"explication (What sources did you use), "
        f"date (exactly '{today_str}')."
    )

    # створюємо агента на кожен запит — безпечно та швидко
    agent = Agent[None, Weather](
        "openai:gpt-4o-mini",
        output_type=Weather,
        instructions=instructions,
    )

    result = await agent.run(query.question)
    weather: Weather = result.output

    return {
        "weather": weather,
        "usage": result.usage(),
    }


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
