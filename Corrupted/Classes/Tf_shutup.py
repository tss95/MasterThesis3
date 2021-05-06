class Tf_shutup():
    
    def __init__(self):
        """
        Make Tensorflow less verbose

        Source:
        https://stackoverflow.com/a/54950981
        """
        try:
            # noinspection PyPackageRequirements
            import os
            from tensorflow import logging
            logging.set_verbosity(logging.ERROR)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
            # noinspection PyUnusedLocal
            def deprecated(date, instructions, warn_once=True):
                def deprecated_wrapper(func):
                    return func
                return deprecated_wrapper

            from tensorflow.python.util import deprecation
            deprecation.deprecated = deprecated

        except ImportError:
            pass