run:
	watchmedo auto-restart --pattern="*.py" --recursive -- python app.py
