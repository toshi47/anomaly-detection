from flask import Flask, jsonify, request
import numpy as np
from rrcf_package.model import Model_RRCF

DDoS_rrcf=Model_RRCF()

app = Flask(__name__)
def init():
    DDoS_rrcf.load_forest('service/models/DDoS/forest_normal.pkl','service/models/DDoS/forest_attack.pkl')
    


@app.route('/predict', methods=['POST'])
def predict_request():  
    print(request.data)
    
    print('rrcf training...')
    # test rrcf
    predictions = rrcf.detection_ddos(X_test)
    for i in range(len(predictions)):
        if predictions[i] == 1:
            print(f"ddos attack")
        else:
            print(f"normal traffic")
    #y_test=DDoS_rrcf.test(X_test, )
    DDoS_rrcf.print_metrics(y_test)
    pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

