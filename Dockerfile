FROM python:3.6-stretch

RUN python --version
RUN pip --version

COPY . ./docker

WORKDIR ./docker

RUN pip install -r requirements.txt
RUN apt update
RUN apt install sshpass

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["new_app.py"]