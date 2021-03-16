from io import BytesIO
from flask import Flask, render_template, request 
# from werkzeug import secure_filename 
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator 
from flask import Flask, jsonify, request 
import numpy as np 
import os 
import base64
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   	tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = tf.keras.models.load_model('D:\\codes\\torch practice\\dog_classification\\resnet_model.h5') 

app = Flask(__name__) 

def finds(): 
	file_path = 'D:\\codes\\torch practice\\dog_classification\\ti.png'
	
	
	vals = ["beagle", "chihuahua", "doberman", "french_bulldog", "golden_retriever", "malamute", "pug", "saint_bernard", "scottish_deerhound", "tibetan_mastiff"] 
	

	from keras.preprocessing.image import img_to_array,load_img
	

	
	im = img_to_array(load_img(file_path, target_size=(224, 224)))

	im = np.expand_dims(im, axis=0)
	pr = model.predict(im)
	score = float(str(max(pr.tolist()[0]))[:5]) * 100
	pr = np.argmax(pr, axis=-1)
	arr = [vals[pr.tolist()[0]], score]
		
	return arr

@app.route('/') 
def upload_f(): 
	arr = finds()
	


	return jsonify({
		'breed': arr[0],
		'score': arr[1]
		
		}) 

 




if __name__ == '__main__': 
	app.run() 
