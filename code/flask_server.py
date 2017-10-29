from flask import Flask
from flask import request
import json
import cloudpickle as pickle
from flask_cors import CORS, cross_origin

with open('finalmodel', 'r') as fp:
	model = pickle.load(fp)

with open('sentences', 'r') as fp:
	sentences = pickle.load(fp)

with open('z', 'r') as fp:
	z = pickle.load(fp)


app = Flask(__name__)
CORS(app, resources={"*" : {"origins" : "*"}})
@app.route('/')
def hello_world():
	try:
		# 
	    return json.dumps({"text" : model.most_similar(request.args.get('query'), sentences, z)[0]})
	except:
		return json.dumps({"text" : 'WRITE SOMETHING'})


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')