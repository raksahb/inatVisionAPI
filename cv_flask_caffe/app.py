import os, classifier, datetime
from flask import Flask, render_template, request, jsonify
from forms import ImageForm
from PIL import Image
import json
import yaml
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

config = yaml.safe_load(open("config.yml"))

CAFFE_MODEL = config["model_file"]
DEPLOY_FILE = config["deploy_file"]
MEAN_FILE = config["mean_file"]
LABELS_FILE = config["labels_file"]
UPLOAD_FOLDER = config["upload_folder"]
TENSORFLOW_MODEL = config["tensorflow_model_file"]

# this is model-specific
def pre_process(filepath):
    size=(224,224)
    im = Image.open(filepath)
    return im.resize(size)

TENSORFLOW_TAXON_IDS = []
with open("taxa.txt") as f:
    for line in f:
        taxon = line.rstrip()
        TENSORFLOW_TAXON_IDS.append(int(taxon))

graph_def = tf.GraphDef( )
with open( TENSORFLOW_MODEL, "rb" ) as f:
    graph_def.ParseFromString( f.read( ) )
tf.import_graph_def( graph_def, name="" )

def doTensor(filepath):
    sess = tf.Session( )
    with sess.as_default( ):
        sess.graph.as_default( )
        graph = sess.graph
        # Load in an image to classify and preprocess it
        image = imread( filepath )
        image = imresize( image, [ 299, 299 ] )
        image = image.astype( np.float32 )
        image = ( image - 128.) / 128.
        image = image.ravel( )
        images = np.expand_dims( image, 0 )
        # Get the input and output operations
        input_op = graph.get_operation_by_name( "images" )
        input_tensor = input_op.outputs[0]
        output_op = graph.get_operation_by_name( "Predictions" )
        output_tensor = output_op.outputs[0]
        # Get the predictions (output of the softmax) for this image
        preds = sess.run(output_tensor, {input_tensor : images})
        sess.close()
        return dict(zip(TENSORFLOW_TAXON_IDS,[ round(elem * 100, 4) for elem in preds[0].astype(float)]))


app = Flask(__name__)
app.debug = True
app.secret_key = config["app_secret"]

@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageForm()
    if request.method == 'POST':
        filepath = form.path.data
        version = form.version.data
        if not filepath:
            image_file = form.image.data
            extension = os.path.splitext(image_file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')) + extension
            image_file.save(filepath)
        if version == 'tensorflow':
            classifications = doTensor(filepath)
        else:
            pre_process(filepath).save(filepath)
            image_files = [filepath]
            classifications = classifier.classify(
                caffemodel=CAFFE_MODEL,
                deploy_file=DEPLOY_FILE,
                image_files=image_files,
                labels_file=LABELS_FILE,
                mean_file=MEAN_FILE,
                use_gpu=False
            )

        return jsonify(classifications)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
