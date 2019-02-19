import csv
import copy

import numpy as np
import tensorflow as tf

class MultiClassInferrer:

    def __init__( self, model_path, mapping_path, options={ } ):
        self.__model_path = model_path
        self.__mapping_path = mapping_path
        # the model has 8 classes. The last class is leaves and is treated
        # differently. The first 7 correspond to the basic taxonomic
        # ranks: Kingdom, Phylum, Class, Order, Family, Genus, Species.
        # This array represents those rank's iNat rank_levels
        self.__task_ranks = [70, 60, 50, 40, 30, 20, 10, "leaves"]
        self.__num_tasks = 8
        self.__load_mapping( )
        self.__prepare_tf_session( options )

    def __load_mapping( self ):
        self.node_key_to_class_id = { }
        self.node_key_to_leaf_class_id = { }
        self.rank_level_class_to_taxon = { }
        for rank_level in self.__task_ranks:
            self.rank_level_class_to_taxon[rank_level] = { }
        self.rank_level_class_to_taxon["leaves"] = { }
        self.taxa = { }
        self.taxon_children = { }
        try:
            with open( self.__mapping_path ) as csv_file:
                csv_reader = csv.DictReader( csv_file, delimiter="," )
                for row in csv_reader:
                    taxon_id = row["taxon_id"]
                    class_id = row["class_id"]
                    rank_level = row["rank_level"]
                    leaf_class_id = row["leaf_class_id"]
                    # create a universal root node with ID LIFE
                    parent = "LIFE" if row["parent_taxon_id"] == "" else row["parent_taxon_id"]
                    self.node_key_to_class_id[taxon_id] = int( class_id )
                    self.rank_level_class_to_taxon[int( rank_level )][class_id] = taxon_id
                    # some taxa are not leaves and aren't represented in the leaf layer
                    if leaf_class_id:
                        self.node_key_to_leaf_class_id[taxon_id] = int( leaf_class_id )
                        self.rank_level_class_to_taxon["leaves"][leaf_class_id] = taxon_id
                    self.taxa[taxon_id] = {
                        "id": taxon_id,
                        "name": row["name"],
                        "class": int( class_id ),
                        "parent": parent,
                    }
                    if parent:
                        if parent not in self.taxon_children:
                            self.taxon_children[parent] = []
                        self.taxon_children[parent].append( taxon_id )
        except IOError as e:
            print( "Cannot open mapping file" )

    def __prepare_tf_session( self, options={} ):
        if "gpu_memory_fraction" not in options:
            options["gpu_memory_fraction"] = 1.0
        tf_session_config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement = True,
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=options["gpu_memory_fraction"]
            )
        )
        try:
            with open( self.__model_path, "rb" ) as f:
                graph_def = tf.GraphDef( )
                graph_def.ParseFromString( f.read( ) )
        except IOError as e:
            print( "Cannot open model file" )
            return
        graph = tf.Graph( )
        with graph.as_default( ):
            tf.import_graph_def( graph_def, name="" )
        self.tf_session = tf.Session( graph=graph, config=tf_session_config )
        self.input_tensor = graph.get_operation_by_name(
            "images" ).outputs[0]
        self.output_tensors = []
        for task in xrange( self.__num_tasks ):
            output_op = graph.get_operation_by_name( "Predictions_task_%d" % task )
            self.output_tensors.append( output_op.outputs[0] )

    def best_path_prediction( self, task_probs, current_id="LIFE", task_index=-1, score_normalizer=1. ):
        if current_id not in self.taxon_children:
            class_probs = task_probs[task_index][0]
            score = class_probs[self.node_key_to_class_id[current_id]] / score_normalizer
            return [[current_id, score]]

        children = self.taxon_children[current_id]
        num_children = len( children )
        child_class_probs = task_probs[task_index+1][0]
        # If the parent has multiple children then we need to determine which child is most likely.
        if num_children > 1:
            # Sort children by probability
            children.sort( key=lambda x: child_class_probs[self.node_key_to_class_id[x]] )
            best_child_id = children[-1]
            child_normalizer = np.sum( map( lambda x: child_class_probs[self.node_key_to_class_id[x]], children ) )
        else:
            # Only one child, so our prediction is real easy
            best_child_id = children[0]
            child_normalizer = child_class_probs[self.node_key_to_class_id[best_child_id]]

        if current_id == "LIFE":
            return self.best_path_prediction( task_probs, best_child_id, task_index + 1, score_normalizer )
        else:
            class_probs = task_probs[task_index][0]
            score = class_probs[self.node_key_to_class_id[current_id]] / score_normalizer
            return [[current_id, score]] + self.best_path_prediction( task_probs, best_child_id, task_index + 1, child_normalizer )

    def best_branch_multiclass( self, task_probs ):
        prediction_path_confs = self.best_path_prediction( task_probs )
        allresults = []
        prev_conf = 1.
        print "Prediction Via Traversing the Taxonomy"
        for node_key, conf in prediction_path_confs:
            rolling_conf = prev_conf * conf
            thisresult = copy.copy( self.taxa[node_key] )
            thisresult["conf"] = float( conf )
            thisresult["score"] = float( rolling_conf )
            parent_id = thisresult["parent"]
            print "\t Node {0:25s} @ conf {1:0.3f} \t(rolling {2:0.3f})".format(
                thisresult["name"], conf, rolling_conf )
            allresults.append( thisresult )
            prev_conf = rolling_conf
        return allresults

    def best_branch_leaves( self, task_probs ):
        leaf_probs = task_probs[7][0]
        aggregated_scores = self.aggreagte_leaf_scores_recursively( leaf_probs )
        return self.build_leaf_branch_resursively( aggregated_scores["all_scores"] )

    def aggreagte_leaf_scores_recursively( self, leaf_probs, current_id="LIFE" ):
        if current_id not in self.taxon_children:
            index = self.node_key_to_leaf_class_id[current_id] if current_id in self.node_key_to_leaf_class_id else 0
            return {
                "score": leaf_probs[index],
                "all_scores": { current_id: leaf_probs[index] }
            }
        all_scores = { }
        children_scores = [ ]
        for child_id in self.taxon_children[current_id]:
            child_scores = ( self.aggreagte_leaf_scores_recursively( leaf_probs, child_id ) )
            all_scores.update( child_scores["all_scores"] )
            children_scores.append( child_scores["score"] )
        this_score = np.sum( children_scores )
        all_scores[current_id] = this_score
        return {
            "score": this_score,
            "all_scores": all_scores
        }

    def build_leaf_branch_resursively( self, all_scores, current_id="LIFE" ):
        if current_id not in self.taxon_children:
            thisresult = copy.copy( self.taxa[current_id] )
            thisresult["score"] = float( all_scores[current_id] )
            return [thisresult]
        children = self.taxon_children[current_id]
        num_children = len( children )
        # If the parent has multiple children then we need to determine which child is most likely.
        if num_children > 1:
            # Sort children by probability
            children.sort( key=lambda x: all_scores[x] )
            best_child_id = children[-1]
        else:
            # Only one child, so our prediction is real easy
            best_child_id = children[0]
        if current_id == "LIFE":
            return self.build_leaf_branch_resursively( all_scores, best_child_id )
        else:
            thisresult = copy.copy( self.taxa[current_id] )
            thisresult["score"] = float( all_scores[current_id] )
            return [thisresult] + self.build_leaf_branch_resursively( all_scores, best_child_id )

    def predictions_by_index( self, index, task_probs, limit=10 ):
        rank_probs = task_probs[index]
        sorted_args = rank_probs[0].argsort( )[::-1][:limit]

        rank_keys = self.rank_level_class_to_taxon[self.__task_ranks[index]]
        rank_results = []
        for arg in sorted_args:
            pred_key = rank_keys[str( arg )]
            rank_result = copy.copy( self.taxa[pred_key] )
            rank_result["score"] = rank_probs[0][arg]
            rank_results.append( rank_result )
        return rank_results

    def predictions_to_json( self, index, task_probs, limit=10 ):
        return dict( {
            arg["id"]: float( round( arg["score"] * 100, 6 ) ) for arg in
              self.predictions_by_index( index, task_probs, limit )
        } )

    def process_images( self, images ):
        return self.tf_session.run( self.output_tensors, { self.input_tensor : images } )
