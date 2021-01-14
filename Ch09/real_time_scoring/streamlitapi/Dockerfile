FROM python:3.7-slim
WORKDIR /streamlit
COPY ./requirements.txt /streamlit
RUN pip install -r requirements.txt
COPY ./webapp.py /streamlit
EXPOSE 8501
CMD ["streamlit", "run", "webapp.py"]
