import numpy as np
import pandas as pd


class Balancer():
    
    def __init__(self, data_loader, handler):
        self.loader = data_loader
        self.handler = handler
        self.seed = data_loader.seed
        
    def downsample_label(self, target_label, name_label_df, n_samples, seed):
        # Downsamples target label
        target_df = name_label_df.loc[name_label_df["label"] == target_label]
        downsampled_target_df = target_df.sample(n_samples, random_state = self.seed)
        
        non_target_df_len = len(name_label_df) - len(target_df) 
        downsampled_df = np.empty((non_target_df_len + len(downsampled_target_df), 2), dtype = '<U32')
        downsampled_df[0:non_target_df_len] = name_label_df.loc[name_label_df["label"] != target_label]
        downsampled_df[non_target_df_len:non_target_df_len + len(downsampled_target_df)] = downsampled_target_df
        downsampled_df = pd.DataFrame(downsampled_df, columns=["name", "label"])
        
        return downsampled_df
        
    def upsample_label(self, target_label, name_label_df, n_samples, seed):
        # Selects n_samples from the target label to include in the new df in addition to the non-target label dps.
        target_df = name_label_df.loc[name_label_df["label"] == target_label]
        
        random_selection = np.empty((len(name_label_df)-len(target_df)+n_samples, 2), dtype = '<U32')
        random_selection[0:len(name_label_df)-len(target_df)] = name_label_df.loc[name_label_df["label"] != target_label]
        
        current_len = len(name_label_df) - len(target_df)
        random_selection[current_len:current_len + n_samples] = target_df.sample(n_samples, replace = True, random_state = seed)
        random_selection = pd.DataFrame(random_selection, columns = ["name", "label"])
        
        return random_selection
    
    def frac_diff_n_samples(self, frac_diff, min_counts, max_counts):
        diff = max_counts - min_counts
        n_samples = int(min_counts + diff*frac_diff)
        return n_samples
        

    def balance_dataset(self, name_label, downsample, upsample, frac_diff = 1):
        """
        Balance the dataset. Downsample, upsample or both.
        
        PARAMETERS:
        ------------------------------
        name_label: np.array - array of all event names and their respective label.
        downsample: bool -     True then will downsample
        upsample:   bool -     True then will upsample such that the length of the least occuring label is 
                               equal to the most occuring
        frac_diff: float -     Fraction of the most prominent label that will be downsampled. 
                               0 will mean that it will be downsampled so that its length is equal 
                               to that of the least occuring label
        
        """
        balancing = pd.DataFrame(name_label, columns = ["name", "label"], dtype='<U32')
        if downsample:
            uniques, counts = self.loader.get_dataset_distribution(np.array(balancing, dtype = '<U32'))
            most_occuring_label = uniques[np.where(counts == max(counts))][0]
            frac_diff_n_samples = self.frac_diff_n_samples(frac_diff, min(counts), max(counts))
            balancing = self.downsample_label(most_occuring_label, balancing, frac_diff_n_samples, self.seed)
                     
        if upsample:
            uniques, counts = self.loader.get_dataset_distribution(np.array(balancing,  dtype = '<U32'))
            least_occuring_label = uniques[np.where(counts == min(counts))][0]
            n_samples_for_balance = max(counts)
            balancing = self.upsample_label(least_occuring_label, balancing, n_samples_for_balance, self.seed)
        balancing = balancing.sample(frac = 1, random_state = self.seed).reset_index(drop=True)
        return np.array(balancing)