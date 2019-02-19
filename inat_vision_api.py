import json
import magic
import numpy as np
import os
import urllib
import uuid
import yaml
import time
import datetime

from flask import Flask, request, jsonify, render_template
from forms import ImageForm
from PIL import Image
from scipy.misc import imread, imresize

from inferrers.single_class_inferrer import SingleClassInferrer
from inferrers.multi_class_inferrer import MultiClassInferrer

CONFIG = yaml.safe_load( open( "config.yml" ) )

class InatVisionAPI:

    def __init__( self ):
        self.setup_legacy_inferrer( )
        self.setup_multiclass_inferrer( )
        self.app = Flask( __name__ )
        self.app.secret_key = CONFIG["app_secret"]
        self.upload_folder = "static/"
        self.app.add_url_rule( "/", "index", self.index_route, methods=["GET", "POST"] )

    def setup_legacy_inferrer( self ):
        model_path = os.path.join( CONFIG["single_class_model_dir"], "optimized_model-3.pb" )
        mapping_path = os.path.join( CONFIG["single_class_model_dir"], "taxa.txt" )
        self.legacy_inferrer = SingleClassInferrer( model_path, mapping_path, {
          "gpu_memory_fraction": 0.45
        }  )

    def setup_multiclass_inferrer( self ):
        model_path = os.path.join( CONFIG["multi_class_model_dir"], "optimized_model.pb" )
        mapping_path = os.path.join( CONFIG["multi_class_model_dir"], "taxonomy_data.csv" )
        self.multiclass_inferrer = MultiClassInferrer( model_path, mapping_path, {
          "gpu_memory_fraction": 0.45
        }  )

    def index_route( self ):
        t = time.time( )
        form = ImageForm( )
        observation_id = request.args["observation_id"] if "observation_id" in request.args else form.observation_id.data
        if request.method == "POST" or observation_id:
            request_start_datetime = datetime.datetime.now( )
            request_start_time = time.time( )
            image_file = form.image.data
            format = request.args["format"] if "format" in request.args else form.format.data
            file_path = None
            image_uuid = None
            if observation_id:
                file_path = self.download_observation( observation_id )
            else:
                file_path = self.process_upload( form.image.data )
            if file_path == None:
                return render_template( "home.html" )

            images = self.prepare_image_for_inference( file_path )
            # Get the multi-class predictions ( output of the softmax ) for this image
            multiclass_predictions = self.multiclass_inferrer.process_images( images )
            best_branch_multiclass = self.multiclass_inferrer.best_branch_multiclass( multiclass_predictions)
            best_branch_multiclass_leaves = self.multiclass_inferrer.best_branch_leaves( multiclass_predictions )
            # Run the image through the original TF model which is loaded into a separate session
            legacy_predictions = self.legacy_inferrer.process_images( images )

            # timing
            dt = time.time( ) - t
            runtime = dt * 1000.
            print( "Execution time: %0.2f" % ( dt * 1000. ))

            render_method = self.render_results_html if format == "html" else self.render_results_json
            return render_method(
              runtime,
              observation_id,
              best_branch_multiclass,
              best_branch_multiclass_leaves,
              multiclass_predictions,
              legacy_predictions )
        else:
            return render_template( "home.html" )

    def render_results_html( self, runtime, observation_id, best_branch_multiclass,
                             best_branch_multiclass_leaves, multiclass_predictions, legacy_predictions ):
        return render_template( "results.html",
            time=runtime,
            observationID=observation_id,
            bestBranch=best_branch_multiclass,
            bestBranchLeaves=best_branch_multiclass_leaves,
            bestSpecies=self.multiclass_inferrer.predictions_by_index( 6, multiclass_predictions, 10 ),
            bestLeaves=self.multiclass_inferrer.predictions_by_index( 7, multiclass_predictions, 10 ),
            legacyResults=self.legacy_inferrer.sorted_predictions( legacy_predictions, 10 ),
            bestKingdoms=self.multiclass_inferrer.predictions_by_index( 0, multiclass_predictions, 10 ),
            bestPhylums=self.multiclass_inferrer.predictions_by_index( 1, multiclass_predictions, 10 ),
            bestClasses=self.multiclass_inferrer.predictions_by_index( 2, multiclass_predictions, 10 ),
            bestOrders=self.multiclass_inferrer.predictions_by_index( 3, multiclass_predictions, 10 ),
            bestFamilies=self.multiclass_inferrer.predictions_by_index( 4, multiclass_predictions, 10 ),
            bestGenera=self.multiclass_inferrer.predictions_by_index( 5, multiclass_predictions, 10 ) )

    def render_results_json( self, runtime, observation_id, best_branch_multiclass,
                             best_branch_multiclass_leaves, multiclass_predictions, legacy_predictions ):
        legacy_results = self.legacy_inferrer.sorted_predictions( legacy_predictions, 500 )
        response_data = {
            "kingdom": self.multiclass_inferrer.predictions_to_json( 0, multiclass_predictions, 500 ),
            "phylum": self.multiclass_inferrer.predictions_to_json( 1, multiclass_predictions, 500 ),
            "class": self.multiclass_inferrer.predictions_to_json( 2, multiclass_predictions, 500 ),
            "order": self.multiclass_inferrer.predictions_to_json( 3, multiclass_predictions, 500 ),
            "family": self.multiclass_inferrer.predictions_to_json( 4, multiclass_predictions, 500 ),
            "genus": self.multiclass_inferrer.predictions_to_json( 5, multiclass_predictions, 500 ),
            "species": self.multiclass_inferrer.predictions_to_json( 6, multiclass_predictions, 500 ),
            "leaf": self.multiclass_inferrer.predictions_to_json( 7, multiclass_predictions, 500 ),
            "legacy": dict( { arg["id"]: round( arg["score"] * 100, 6 ) for arg in legacy_results } ),
            "bestBranch": best_branch_multiclass,
            "bestBranchFromLeaves": best_branch_multiclass_leaves
        }
        return jsonify( response_data )

    def process_upload( self, form_image_data ):
        if form_image_data == None:
            return None
        extension = os.path.splitext( form_image_data.filename )[1]
        image_uuid = str( uuid.uuid4( ) )
        file_path = os.path.join( self.upload_folder, image_uuid ) + extension
        form_image_data.save( file_path )
        return file_path

    def download_observation( self, observation_id ):
        url = "https://api.inaturalist.org/v1/observations/" + observation_id
        cache_path = os.path.join( self.upload_folder, "downloaded-obs-" ) + observation_id + ".jpg"
        if os.path.exists( cache_path ):
            return cache_path
        response = urllib.urlopen( url )
        data = json.loads( response.read( ) )
        if data == None or data["results"] == None or data["results"][0] == None or data["results"][0]["photos"] == None or data["results"][0]["photos"][0] == None or data["results"][0]["photos"][0]["url"] == None:
            return None
        urllib.urlretrieve( data["results"][0]["photos"][0]["url"].replace( "square", "medium" ), cache_path )
        return cache_path

    def prepare_image_for_inference( self, file_path ):
        mime_type = magic.from_file( file_path, mime=True )
        # attempt to convert non jpegs
        if mime_type != "image/jpeg":
            im = Image.open( file_path )
            rgb_im = im.convert( "RGB" )
            file_path = os.path.join( self.upload_folder, image_uuid ) + ".jpg"
            rgb_im.save( file_path )

        # Load in an image to classify and preprocess it
        image_fp = file_path
        image = imread( image_fp )
        image = imresize( image, [299, 299] )
        image = image.astype( np.float32 )
        image = ( (image / 255. )  - 0.5 ) * 2.
        return np.expand_dims( image, 0 )
