.PHONY: server agent client

server:
	cd server && poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

agent:
	poetry run python agent/executor.py --server-url http://localhost:8000 --port 8010

client:
	cd client && npm run dev
