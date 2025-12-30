from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.chat import router as chat_router
from core.config import config
from core.logging import setup_logging

app = FastAPI(title=config.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    setup_logging()


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat_router, prefix=config.API_STR)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
