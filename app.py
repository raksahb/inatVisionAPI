import datetime
import json
import magic
import os
import random
import time
import uuid
import yaml
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from forms import ImageForm
from PIL import Image
from scipy.misc import imread, imresize

config = yaml.safe_load(open("config.yml"))

def _load_taxon_ids(taxa_file):
    taxon_ids = []
    with open(taxa_file) as f:
        for line in f:
            iter, taxon_id = line.rstrip().split(": ")
            taxon_ids.append(int(taxon_id))
    return taxon_ids
TENSORFLOW_TAXON_IDS = _load_taxon_ids("taxa.txt")

app = Flask(__name__)
app.secret_key = config["app_secret"]

UPLOAD_FOLDER = "static/"

graph = None
sess = tf.Session()
with sess.as_default():
    # Load in the graph
    graph_def = tf.GraphDef()
    with open('optimized_model-3.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph = sess.graph

sess.graph.finalize()

# Get the input and output operations
input_op = graph.get_operation_by_name('images')
input_tensor = input_op.outputs[0]
output_op = graph.get_operation_by_name('Predictions')
output_tensor = output_op.outputs[0]

def write_logstash(image_file, image_uuid, file_path, request_start_datetime, request_start_time, mime_type):
    request_end_time = time.time()
    request_time = round((request_end_time - request_start_time) * 1000, 6)
    logstash_log = open('log/logstash.log', 'a')
    log_data = {'@timestamp': request_start_datetime.isoformat(),
                'uuid': image_uuid,
                'duration': request_time,
                'mime_type': mime_type,
                'client_ip': request.access_route[0],
                'filename': image_file.filename,
                'image_size': os.path.getsize(file_path)}
    json.dump(log_data, logstash_log)
    logstash_log.write("\n")
    logstash_log.close()

@app.route('/', methods=['GET', 'POST'])
def classify():
    form = ImageForm()
    if request.method == 'POST':
        request_start_datetime = datetime.datetime.now()
        request_start_time = time.time()
        image_file = form.image.data
        extension = os.path.splitext(image_file.filename)[1]
        image_uuid = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + extension
        image_file.save(file_path)

        mime_type = magic.from_file(file_path, mime=True)
        # attempt to convert non jpegs
        if mime_type != 'image/jpeg':
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
            file_path = os.path.join(UPLOAD_FOLDER, image_uuid) + '.jpg'
            rgb_im.save(file_path)

        # Load in an image to classify and preprocess it
        # Note that we are using imread to convert to RGB in case the image was
        # in grayscale or something: https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html
        # Also note that imread is deprecated and we should probably switch to
        # imageio, and/or use PIL to perform this RGB conversion ourselves
        image = imread(file_path, False, 'RGB')
        image = imresize(image, [299, 299])
        image = image.astype(np.float32)
        image = (image - 128.) / 128.
        image = image.ravel()
        images = np.expand_dims(image, 0)

        # Get the predictions (output of the softmax) for this image
        preds = sess.run(output_tensor, {input_tensor : images})

        sorted_pred_args = preds[0].argsort()[::-1][:100]
        response_json = jsonify(dict({TENSORFLOW_TAXON_IDS[arg]: round(preds[0][arg] * 100, 6) for arg in sorted_pred_args}))
        write_logstash(image_file, image_uuid, file_path, request_start_datetime, request_start_time, mime_type)
        return response_json
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
