"""Start the NER FastAPI server."""
import uvicorn
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config()
    api_config = config["api"]
    uvicorn.run("src.api.serve:app", host=api_config["host"], port=api_config["port"], reload=False)
