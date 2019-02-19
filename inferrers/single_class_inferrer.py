import tensorflow as tf

class SingleClassInferrer:

    def __init__( self, model_path, mapping_path, options={ } ):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.__load_mapping( )
        self.__prepare_tf_session( options )

    def __load_mapping( self ):
        self.mappings = [ ]
        with open( self.mapping_path ) as f:
            for line in f:
                iter, mapped_id = line.rstrip( ).split( ": " )
                self.mappings.append( int( mapped_id ) )

    def __prepare_tf_session( self, options={ } ):
        if "gpu_memory_fraction" not in options:
            options["gpu_memory_fraction"] = 1.0
        tf_session_config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=options["gpu_memory_fraction"]
            )
        )
        self.tf_session = tf.Session( config=tf_session_config )
        with self.tf_session.as_default( ):
            # Load in the graph
            graph_def = tf.GraphDef( )
            with open( self.model_path, "rb" ) as f:
                graph_def.ParseFromString( f.read( ) )
            self.tf_session.graph.as_default( )
            tf.import_graph_def( graph_def, name="" )
        self.tf_session.graph.finalize( )
        self.input_tensor = self.tf_session.graph.get_operation_by_name(
            "images" ).outputs[0]
        self.output_tensor = self.tf_session.graph.get_operation_by_name(
            "Predictions" ).outputs[0]

    def mapped_id( self, index ):
        return self.mappings[index]

    def process_images( self, images ):
        return self.tf_session.run( self.output_tensor, { self.input_tensor : images } )

    def sorted_predictions( self, predictions, limit=10):
        sorted_pred_args = predictions[0].argsort( )[::-1][:limit]
        results = []
        for arg in sorted_pred_args:
            speciesResult = {
                "id": self.mapped_id( arg ),
                "score": predictions[0][arg]
            }
            results.append( speciesResult )
        return results
