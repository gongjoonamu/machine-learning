from flask import Flask, request, jsonify, abort
from flasgger import Swagger
from predict import Predict

app = Flask(__name__)
swagger = Swagger(app)  # Swagger docs served at /apidocs


@app.route("/predict/<int:question>", methods=['POST'])
def predict(question):
    '''
    Predicts the grade for a given conversation question
    ---
    parameters:
        - name: question
          in: path
          type: int
          required: true
    responses:
        200:
            description: A grade for the conversations
    '''
    if not request.is_json or question not in range(1, 4):
        abort(500)
    else:
        content = request.get_json()
        msg = content.get('msg')
        prediction = Predict(question)
        return jsonify(prediction.run(msg))


@app.route('/health', methods=['GET'])
def health():
    '''
    Health check
    ---
    responses:
        200:
            description: Successful health check response
    '''
    return 'OK!'


@app.route('/', methods=['GET'])
def root():
    return 'OK!'


if __name__ == '__main__':
    app.run(host='0.0.0.0')
