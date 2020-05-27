import flask
from flask import request, jsonify, render_template, make_response
import numpy as np 
import cv2
import sys
import base64
import io
from knn import Knn
from PIL import Image
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

knn = Knn()
knn.set_k(70)


@app.route('/getMathAnswer', methods=['POST'])
def getMathAnswer():
	if 'base64' in request.json:
		try:
			data = request.json['base64']
			imgdata = base64.b64decode(data)
			filename = 'some_image.png'
			# print(request.json)
			with open(filename, 'wb') as f:
			    f.write(imgdata)
			im = cv2.imread(filename)
			# customHeight = int(request.json['height'])
			# customHeight = int(customHeight*0.7)
			# print(customHeight)
			im = cv2.resize(im,(int(request.json['width']),int(request.json['height'])), interpolation = cv2.INTER_AREA)
			gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			# (2) threshold-inv and morph-open 
			th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
			morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2)))
			# (3) find and filter contours, then draw on src 
			cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
			res = ''

			for cnt in cnts:
				x,y,w,h = bbox = cv2.boundingRect(cnt)
				if  h>28:
					cv2.rectangle(cnt, (x,y), (x+w, y+h), (255, 0, 255), 1, cv2.LINE_AA)
					roi = threshed[y:y+h,x:x+w]
					roi = center_image(roi)
					roismall = cv2.resize(roi,(28,28))
					# cv2.imshow( "Display window", roismall )
					# cv2.waitKey(0);
					roismall = roismall.flatten()
					predict = knn.predict(roismall)
					res += str(predict)
			# print(res)
			return res
		except:
			return jsonify(status= 'Image processing failed')
	else:
		return jsonify(status= 'Not image fail')



def center_image(im):
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 150
    border = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    return border


def addHeaders(response):
    response.headers.add('Content-Type', 'application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'PUT, GET, POST, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Content-Length,Authorization,X-Pagination')
    return response

if __name__ == '__main__':
    # app.run(port = 5000, debug=True)
	app.run(host= '0.0.0.0')