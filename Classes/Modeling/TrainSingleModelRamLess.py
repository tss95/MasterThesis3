
import gc
import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.DataProcessing.RamLessGen import RamLessGen
from Classes.Modeling.TrainSingleModel import TrainSingleModel

class TrainSingleModelRamLess(TrainSingleModel):

    """
    Responsible for training a model when data cannot be held in RAM. Also performs all necessary processes related to storing of the model
    and initializing relevant processes.

    PARAMETERS:
    -------------------------------------------------------------------------------
    ramLessLoader: (object)      Fitted ramLessLoader. Responsible to transforming the data through the generator.
    helper: (object)             HelperFunctions object. Holds functions which are used for many processes.
    loadData: (object)           Fitted LoadData object.
    model_type: (str)            String representing the name of the model architecture to be trained.
    num_channels: (int)          Option to train and evaluate the models on a reduced number of channels. P-beam is the last channel to be removed.
    use_tensorboard: (bool)      Whether or not to log to tensorboard. Does not launch tensorboard.
    use_liveplots: (bool)        Whether or not to use LiveLossPlots. Requires Keras to be installed along with LiveLossPLots.
    use_custom_callback: (bool)  Whether or not to use custom_callback. Required to get FBeta after each epoch. Will log FBeta to results file if log_data == True.
    use_early_stopping: (bool)   Whether or not to use early stopping. Default parameters. Parameters can be changed in HelperFunctions.py.
    use_reduced_lr: (bool)       Whether or not to use reduce learning rate on plateau with default parameters. Can be changed in HelperFunctions.py.
    log_data: (bool)             Whether or not to log the results to the results file.
    start_from_scratch: (bool)   Whether or not to erease the results file prior to training. Has not been used for many iterations of the code. May not work as expected.
    beta: (float)                Beta value to use for FBeta.
    """
    
    def __init__(self, ramLessLoader, helper, loadData, 
                 model_type, num_channels, use_tensorboard, use_liveplots, use_custom_callback,
                 use_early_stopping, use_reduced_lr, log_data = True, start_from_scratch = False, beta = 1):
            self.ramLessLoader = ramLessLoader        
            self.helper = helper
            self.loadData = loadData
            self.model_type = model_type
            self.num_channels = num_channels
            self.use_tensorboard  = use_tensorboard
            self.use_liveplots = use_liveplots
            self.use_custom_callback = use_custom_callback
            self.use_early_stopping = use_early_stopping
            self.use_reduced_lr = use_reduced_lr
            self.log_data = log_data
            self.start_from_scratch = start_from_scratch
            self.beta = beta
            self.num_classes = len(set(self.loadData.label_dict.values()))
            self.resultsProcessor = self.resultsProcessor = GridSearchResultProcessor(self.num_classes, self.model_type, self.loadData, self.ramLessLoader, self.use_early_stopping, self.num_channels, self.beta)
            super().__init__(self.resultsProcessor)

    
    def prep_and_fit_model(self, model, ramLessLoader, workers, max_queue_size, meier_mode = False, **p):
        norm_scale = False
        if ramLessLoader.scaler_name == "normalize":
            norm_scale = True
        self.gen_args = {"batch_size" : p["batch_size"],
                         "ramLessLoader" : ramLessLoader,
                         "num_channels" : self.num_channels,
                         "norm_scale" : norm_scale,
                         "shuffle" : False}
        train_gen = RamLessGen(ramLessLoader.train_ds, ramLessLoader.y_train, ramLessLoader.train_timeAug, **self.gen_args)
        val_gen = RamLessGen(ramLessLoader.val_ds, ramLessLoader.y_val, ramLessLoader.val_timeAug, **self.gen_args)
        return self.fit_model(model, train_gen, val_gen, ramLessLoader.y_val, workers, max_queue_size, meier_mode, **p)


    def metrics_producer(self, model, ramLessLoader, workers, max_queue_size, meier_mode = False,**p):
        val_gen = RamLessGen(ramLessLoader.val_ds, ramLessLoader.y_val, ramLessLoader.val_timeAug, **self.gen_args)
        val_eval = model.evaluate(x = val_gen, 
                                  batch_size = p["batch_size"],
                                  steps = self.helper.get_steps_per_epoch(self.loadData.val, p["batch_size"]),
                                  return_dict = True)
        del val_gen
        eval_args = {"batch_size": p["batch_size"],
                     "label_dict" : self.loadData.label_dict,
                     "num_channels" : self.num_channels,
                     "ramLessLoader" : ramLessLoader,
                     "num_classes" : self.num_classes,
                     "beta" : self.beta}
        val_conf, _, val_acc, val_precision, val_recall, val_fscore = self.helper.evaluate_RamLessGenerator(model, ramLessLoader.val_ds, ramLessLoader.y_val, ramLessLoader.val_timeAug, **eval_args)       
        metrics = {}

        metrics['val'] = {  "val_loss" : val_eval["loss"],
                            "val_accuracy" : val_acc,
                            "val_precision": val_precision,
                            "val_recall" : val_recall,
                            f"val_f{self.beta}" : val_fscore}

        
        print("Evaluating train:")
        train_gen = RamLessGen(ramLessLoader.train_ds, 
                               ramLessLoader.y_train, 
                               p["batch_size"], ramLessLoader, 
                               ramLessLoader.train_timeAug, 
                               self.num_channels, 
                               norm_scale = norm_scale, 
                               shuffle = False)
        train_eval = model.evaluate(x = train_gen, 
                                    batch_size = p["batch_size"],
                                    steps = self.helper.get_steps_per_epoch(self.loadData.train, p["batch_size"]),
                                    return_dict = True)
        del train_gen
        _, _, train_acc, train_precision, train_recall, train_fscore = self.helper.evaluate_RamLessGenerator(model, ramLessLoader.train_ds, ramLessLoader.y_train, ramLessLoader.train_timeAug, **eval_args)
        metrics['train'] = { "train_loss" : train_eval["loss"],
                            "train_accuracy" : train_acc,
                            "train_precision": train_precision,
                            "train_recall" : train_recall,
                            f"train_f{self.beta}" : train_fscore}
        return metrics, val_conf
    
    def run(self, ramLessLoader, workers, max_queue_size, evaluate_train = False, evaluate_val = False, evaluate_test = False, meier_mode = False, index = None, **p):
        if self.log_data and self.results_df is None and self.results_file_name is None:
            self.p = p
            self.create_result_file()
        model = self.create_and_compile_model(ramLessLoader.input_shape, index = index, meier_mode = meier_mode, **p)
        model = self.prep_and_fit_model(model, ramLessLoader, workers, max_queue_size, meier_mode = meier_mode, **p)
        if self.log_data and self.results_df is not None and self.results_file_name != None:
            metrics, val_conf = self.metrics_producer(model, ramLessLoader, max(1, int(workers//2)), int(max_queue_size), meier_mode, **p)
            self.results_df = self.resultsProcessor.store_metrics_after_fit(metrics, val_conf, self.results_df, self.results_file_name)
            print(self.results_df.iloc[-1])
        if evaluate_train:
            print("Unsaved train eval:")
            self.helper.evaluate_RamLessGenerator(model, ramLessLoader.train_ds, ramLessLoader.y_train, p["batch_size"], self.loadData.label_dict, self.num_channels, ramLessLoader, ramLessLoader.noiseAug, self.num_classes, plot_confusion_matrix = True, plot_p_r_curve = True, beta = self.beta)
        if evaluate_val:
            print("Unsaved val eval:")
            self.helper.evaluate_RamLessGenerator(model, ramLessLoader.val_ds, ramLessLoader.y_val, p["batch_size"], self.loadData.label_dict, self.num_channels, ramLessLoader, ramLessLoader.noiseAug, self.num_classes, plot_confusion_matrix = True, plot_p_r_curve = True, beta = self.beta)
        if evaluate_test:
            print("Unsaved test eval:")
            self.helper.evaluate_RamLessGenerator(model, ramLessLoader.test_ds, ramLessLoader.y_test, p["batch_size"], self.loadData.label_dict, self.num_channels, ramLessLoader, ramLessLoader.noiseAug, self.num_classes, plot_confusion_matrix = True, plot_p_r_curve = True, beta = self.beta)
        gc.collect()
        return model, self.results_df