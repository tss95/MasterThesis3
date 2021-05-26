import gc
import traceback
import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)
from Classes.Modeling.DynamicModels import DynamicModels
import pprint

class TrainSingleModel():

    """
    Parent class of TrainSingleModelRam and TrainSingleModelRamLess. Holds functions which both can utilize as is.

    Note: This class should never be initiated directly.

    PARAMETERS:
    ----------------------------------------------------------------
    resultProcessor: (object)           GridSearchResultProcessor object, already initialized.
    """
    
    def __init__(self, resultProcessor):
        self.results_df = None
        self.results_file_name = None        
        self.resultsProcess = resultProcessor
        
        

    def create_result_file(self):
        print("Trying to create result file")
        if self.log_data and self.results_df is None and self.results_file_name is None:
            self.results_file_name = self.resultsProcessor.get_results_file_name()
            self.results_df = self.resultsProcessor.initiate_results_df_opti(self.results_file_name, self.num_classes, self.start_from_scratch, self.p)
            print("Made result file: ", self.results_file_name)
        
    def create_and_compile_model(self, input_shape, index = None, meier_mode = False, **p):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
        tf.config.optimizer.set_jit(True)
        mixed_precision.set_global_policy('mixed_float16')
        p = self.helper.handle_hyperparams(self.num_classes, **p)
        
        if index != None:
            model_info = {"model_type" : self.model_type, "index" : index}
        else:
            model_info = {"model_type" : self.model_type}
        current_picks = [model_info, p]
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(current_picks)
        if self.log_data and self.results_df is not None and self.results_file_name != None:
            self.results_df = self.resultsProcessor.store_params_before_fit_opti(p, self.results_df, self.results_file_name)
        
        
        model = DynamicModels(self.model_type, self.num_classes, input_shape, **p).model
        if not meier_mode:
            opt = self.helper.get_optimizer(p["optimizer"], p["learning_rate"])
            model_compile_args = self.helper.generate_model_compile_args(opt, self.num_classes, self.loadData.balance_non_train_set, self.loadData.noise_not_noise)
            model.compile(**model_compile_args)
        return model
        
    def metrics_dict(self, metrics, set_name, loss, accuracy, precision, recall, fscore, beta):
        metrics[set_name] = {f"{set_name}_loss" : loss,
                             f"{set_name}_accuracy" : accuracy,
                             f"{set_name}_precision" : precision,
                             f"{set_name}_recall" : recall,
                             f"{set_name}_f{beta}" : fscore}
        return metrics
        
    def fit_model(self, model, train_gen, val_gen, y_val, workers, max_queue_size, meier_mode = False, **p):
        if meier_mode:
            fit_args = self.helper.generate_meier_fit_args(self.loadData.train, self.loadData.val, self.loadData,
                                                            p["batch_size"], p["epochs"], val_gen, workers, max_queue_size,
                                                            use_tensorboard = self.use_tensorboard, 
                                                            use_liveplots = self.use_liveplots, 
                                                            use_custom_callback = self.use_custom_callback,
                                                            use_early_stopping = self.use_early_stopping,
                                                            use_reduced_lr = self.use_reduced_lr, y_val = y_val,  beta = self.beta)
        else:    
            fit_args = self.helper.generate_fit_args(self.loadData.train, self.loadData.val, self.loadData,
                                                    p["batch_size"], p["epochs"], val_gen, workers, max_queue_size,
                                                    use_tensorboard = self.use_tensorboard, 
                                                    use_liveplots = self.use_liveplots, 
                                                    use_custom_callback = self.use_custom_callback,
                                                    use_early_stopping = self.use_early_stopping,
                                                    use_reduced_lr = self.use_reduced_lr,
                                                    y_val = y_val, beta = self.beta)
    
        print(f"Utilizes {self.helper.get_steps_per_epoch(self.loadData.val, p['batch_size'])*p['batch_size']}/{len(self.loadData.val)} validation points")
        print(f"Utilizes {self.helper.get_steps_per_epoch(self.loadData.train, p['batch_size'])*p['batch_size']}/{len(self.loadData.train)} training points")
        print("-------------------------------------------------------------------")
        # Fit the model using the generated args
        try:
            # Try block allows for evaluation of the current state of the model at any time.
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()
            tf.config.optimizer.set_jit(True)
            mixed_precision.set_global_policy('mixed_float16')
            model.fit(train_gen, **fit_args)
        except Exception:
            traceback.print_exc()
            #model = None
        finally:
            del train_gen, val_gen 
            gc.collect()   
            return model
