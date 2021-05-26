
import gc
import os
base_dir = '/media/tord/T7/Thesis_ssd/MasterThesis3'
os.chdir(base_dir)
from Classes.Modeling.TrainSingleModel import TrainSingleModel
from Classes.Modeling.GridSearchResultProcessor import GridSearchResultProcessor
from Classes.DataProcessing.RamGen import RamGen


class TrainSingleModelRam(TrainSingleModel):
    """
    Responsible for training a model when data can be held in RAM. Also performs all necessary processes related to storing of the model
    and initializing relevant processes.

    PARAMETERS:
    -------------------------------------------------------------------------------
    noiseAug: (object)          Fitted noise augmentor. Can be None if it is not used.
    helper: (object)            HelperFunctions object. Holds functions which are used for many processes.
    loadData: (object)           Fitted LoadData object.
    model_type: (str)            String representing the name of the model architecture to be trained.
    num_channels: (int)          Option to train and evaluate the models on a reduced number of channels. P-beam is the last channel to be removed.
    use_tensorboard: (bool)      Whether or not to log to tensorboard. Does not launch tensorboard.
    use_liveplots: (bool)        Whether or not to use LiveLossPlots. Requires Keras to be installed along with LiveLossPLots.
    use_custom_callback: (bool)  Whether or not to use custom_callback. Required to get FBeta after each epoch. Will log FBeta to results file if log_data == True.
    use_early_stopping: (bool)   Whether or not to use early stopping. Default parameters. Parameters can be changed in HelperFunctions.py.
    use_reduced_lr: (bool)       Whether or not to use reduce learning rate on plateau with default parameters. Can be changed in HelperFunctions.py.
    ramLoader: (object)         The ramLoader or ramLessLoader object. Used to get preprocessing hyperparameters.
    log_data: (bool)             Whether or not to log the results to the results file.
    start_from_scratch: (bool)   Whether or not to erease the results file prior to training. Has not been used for many iterations of the code. May not work as expected.
    beta: (float)                Beta value to use for FBeta.
    """

    
    def __init__(self, noiseAug, helper, loadData, 
                 model_type, num_channels, use_tensorboard, use_liveplots, use_custom_callback,
                 use_early_stopping, use_reduced_lr, ramLoader, log_data = True, start_from_scratch = False, beta = 1):        
        self.num_channels = num_channels
        self.beta = beta
        
        self.noiseAug = noiseAug
        self.helper = helper
        self.loadData = loadData
        self.ramLoader = ramLoader
        
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.use_reduced_lr = use_reduced_lr
        
        self.num_classes = len(set(self.loadData.label_dict.values()))
        
        self.model_type = model_type
        self.log_data = log_data
        self.start_from_scratch = start_from_scratch
        self.resultsProcessor = GridSearchResultProcessor(self.num_classes, self.model_type, self.loadData, self.ramLoader, self.use_early_stopping, self.num_channels, self.beta)
        super().__init__(self.resultsProcessor)
    
    def prep_and_fit_model(self, model, x_train, y_train, x_val, y_val, workers, max_queue_size, meier_mode = False, **p):
        norm_scale = False
        if self.ramLoader.scaler_name == "normalize":
            norm_scale = True
        self.gen_args = {"batch_size" : p["batch_size"],
                        "noiseAug" : self.noiseAug,
                        "num_channels" : self.num_channels,
                        "norm_scale" : norm_scale,
                        "label_dict" : self.loadData.label_dict}
        
        train_gen = RamGen(traces = x_train, labels = y_train, **self.gen_args)
        val_gen = RamGen(traces = x_val, labels = y_val, **self.gen_args)

        return self.fit_model(model, train_gen, val_gen, y_val, workers, max_queue_size, meier_mode, **p)


    def metrics_producer(self, model, x_train, y_train, x_val, y_val, workers, max_queue_size, meier_mode = False,**p):
        val_gen = RamGen(traces = x_val, labels = y_val, **self.gen_args)
        val_eval = model.evaluate(x = val_gen, batch_size = p["batch_size"],
                                  steps = self.helper.get_steps_per_epoch(self.loadData.val, p["batch_size"]),
                                  return_dict = True)
        eval_args = {"batch_size" : p["batch_size"],
                     "label_dict" : self.loadData.label_dict,
                     "num_channels" : self.num_channels,
                     "noiseAug" : self.noiseAug,
                     "scaler_name" : self.ramLoader.scaler_name,
                     "num_classes" : self.num_classes,
                     "beta" : self.beta}
        del val_gen
        val_conf, _, val_acc, val_precision, val_recall, val_fscore = self.helper.evaluate_generator(model, x_val, y_val, **eval_args)       
        metrics = {}
        metrics = self.metrics_dict(metrics, "val", val_eval["loss"], val_acc, val_precision, val_recall, val_fscore, self.beta)
        print("Evaluating train:")
        train_gen = RamGen(traces = x_train, labels = y_train, **self.gen_args)
        train_eval = model.evaluate(x = train_gen, batch_size = p["batch_size"],
                                    steps = self.helper.get_steps_per_epoch(self.loadData.train, p["batch_size"]),
                                    return_dict = True)
        del train_gen
        _, _, train_acc, train_precision, train_recall, train_fscore = self.helper.evaluate_generator(model, x_train, y_train, **eval_args)
        metrics = self.metrics_dict(metrics, "train", train_eval["loss"], train_acc, train_precision, train_recall, train_fscore, self.beta)
        return metrics, val_conf
    
    def run(self, x_train, y_train, x_val, y_val, x_test, y_test, workers, max_queue_size, evaluate_train = False, evaluate_val = False, evaluate_test = False, meier_mode = False, index = None, **p):
        if self.log_data:
            self.p = p
            self.create_result_file()
        input_shape = (x_train.shape[2], x_train.shape[1])
        model = self.create_and_compile_model(input_shape, index = index, meier_mode = meier_mode, **p)
        model = self.prep_and_fit_model(model, x_train, y_train, x_val, y_val, workers, max_queue_size, meier_mode = meier_mode, **p)
        if self.log_data:
            metrics, val_conf = self.metrics_producer(model, x_train, y_train, x_val, y_val, max(1, int(workers//2)), int(max_queue_size), meier_mode, **p)
            self.results_df = self.resultsProcessor.store_metrics_after_fit(metrics, val_conf, self.results_df, self.results_file_name)
            print(self.results_df.iloc[-1])
        if evaluate_train:
            print("Unsaved train eval:")
            self.helper.evaluate_generator(model, x_train, y_train, p['batch_size'], self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name, self.num_classes, beta = self.beta)
        if evaluate_val:
            print("Unsaved val eval:")
            self.helper.evaluate_generator(model, x_val, y_val, p["batch_size"],self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name,  self.num_classes, beta = self.beta)
        if evaluate_test:
            print("Unsaved test eval:")
            self.helper.evaluate_generator(model, x_test, y_test, p["batch_size"], self.loadData.label_dict, self.num_channels, self.noiseAug, self.ramLoader.scaler_name,  self.num_classes, beta = self.beta)
        gc.collect()
        return model