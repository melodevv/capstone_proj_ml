from flask import Flask, render_template, request, json
from IxiasBot.ixiasai import chatbot_response

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    start = True
    print("Start chatting to Ixias.")
    while start:
        try:
            res = chatbot_response(text)
            return res
        except:
            return "Oops! Did not catch your statment, please rephrase."


if __name__ == "__main__":
    app.run()
