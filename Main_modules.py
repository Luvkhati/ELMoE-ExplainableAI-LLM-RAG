# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:48:51 2025

@author: arnab
"""


import pandas as pd
import re
import ast
import numpy as np
from typing import Dict, List

from Evaluation import Evaluation_metrics
import LLM_API_calls
from LLM_API_calls import Together_models
from Prompt_generator_module import Prompt_generator
from Trained_ML_model_functions import Load_ModelFile, Load_Scalers
from XAI_benchmarks import Lime_wrapper_function, Dice_wrapper_function, Shap_feature_attributions

from Cat_variable_functions import fit_cat_encoder, transform_cat_encoder, Encode_all, Decode_all

from Trained_ML_model_functions import get_model_predictions_classification

#New function to extract 2 test datasets from a bigger datapool without replacement
def Extract_random_test_dataset(dataframe, number_of_test_points):
    assert  number_of_test_points <= len(dataframe), "Sample sizes exceed available data."

    # Sample first set
    df1 = dataframe.sample(n= number_of_test_points, replace=False, random_state=42)

    # Sample second set from remaining data
    # remaining = dataframe.drop(df1.index)
    # df2 = remaining.sample(n=len2, replace=False, random_state=24)
    
    return df1


def Sample_test_dataset(
    test_df: pd.DataFrame,
    training_df: pd.DataFrame,
    X_labels: list,
    len1: int,
    len2: int,
    mode: str = 'bin_quantile_sampling',
    X_cat_encoder=None,
    Y_cat_encoder=None,
    n_bins: int = 10,
    *,
    standardize: bool = True,
    random_state: int = 42,
    cap_high_q: float | None = 0.95,     # cap upper tail (None disables)
    cap_low_q: float | None = 0.0,       # cap lower tail (None disables)
    mixture_alpha: float = 0.30,         # for 'mixture_sampling': fraction from binned; rest from MC
) -> pd.DataFrame:
    """
    Sample a subset of the test dataframe based on train characteristics.

    Modes
    -----
    - 'outlier_sampling':
        Select top-k largest distances (within capped region if enabled).
    - 'bin_quantile_sampling':
        Stratify by distance quantile bins (within capped region if enabled) and sample ~uniformly per bin.
    - 'mixture_sampling':
        (1 - mixture_alpha) fraction via Monte Carlo (uniform row sampling) within capped region,
        mixture_alpha fraction via bin_quantile_sampling within capped region. Then combine + dedupe + top-up.

    Capping
    -------
    If cap_high_q/cap_low_q are set, only consider candidates whose distance lies within
    [quantile(cap_low_q), quantile(cap_high_q)] of the *test* distance distribution (computed vs training stats).
    This avoids extreme tails dominating the selected subset.
    """

    assert len1 + len2 <= len(test_df), "Sample sizes exceed available data."
    if not (0.0 <= mixture_alpha <= 1.0):
        raise ValueError("mixture_alpha must be in [0, 1].")

    rng = np.random.default_rng(random_state)

    # --- encode categorical if provided ---
    if Y_cat_encoder is not None:
        test_dataframe = Encode_all(test_df.copy(), X_cat_encoder, Y_cat_encoder)
        training_df_enc = Encode_all(training_df.copy(), X_cat_encoder, Y_cat_encoder)
    else:
        test_dataframe = test_df
        training_df_enc = training_df

    # --- compute distances ---
    train_X = training_df_enc[X_labels].copy()
    mean_vec = train_X.mean(axis=0)

    test_X = test_dataframe[X_labels].copy()
    if standardize:
        std_vec = train_X.std(axis=0).replace(0, 1.0)
        Z = (test_X - mean_vec) / std_vec
        dists = np.sqrt((Z ** 2).sum(axis=1))
    else:
        dists = np.sqrt(((test_X - mean_vec) ** 2).sum(axis=1))

    dists = pd.Series(dists, index=test_dataframe.index, name="dist")

    # --- cap extremes (build candidate pool) ---
    candidate_idx = dists.index

    if cap_low_q is not None and cap_low_q > 0.0:
        lo = dists.quantile(cap_low_q)
        candidate_idx = candidate_idx[dists.loc[candidate_idx] >= lo]

    if cap_high_q is not None and cap_high_q < 1.0:
        hi = dists.quantile(cap_high_q)
        candidate_idx = candidate_idx[dists.loc[candidate_idx] <= hi]

    if len(candidate_idx) < len2:
        raise ValueError(
            f"After capping, only {len(candidate_idx)} candidates remain, but len2={len2}. "
            f"Relax cap_high_q/cap_low_q or reduce len2."
        )

    d_cand = dists.loc[candidate_idx]

    # -------- helpers (operate within candidate pool) --------
    def _sample_binned(d: pd.Series, k: int) -> pd.Index:
        """Quantile-bin uniform sampling within d.index."""
        if k <= 0:
            return pd.Index([])

        bins = pd.qcut(d, q=n_bins, duplicates="drop")
        used_bins = bins.cat.categories
        B = len(used_bins)
        if B == 0:
            raise ValueError("Not enough distance variation to create bins (after capping).")

        base = k // B
        rem = k % B

        picked = []
        for i, cat in enumerate(used_bins):
            idx_in_bin = d.index[bins == cat].to_numpy()
            take = base + (1 if i < rem else 0)
            if take <= 0:
                continue
            if len(idx_in_bin) <= take:
                chosen = idx_in_bin
            else:
                chosen = rng.choice(idx_in_bin, size=take, replace=False)
            picked.extend(chosen.tolist())

        picked = pd.Index(picked).unique()

        # top-up if some bins were small / duplicates removed
        if len(picked) < k:
            remaining = d.index.difference(picked)
            need = k - len(picked)
            extra = rng.choice(remaining.to_numpy(), size=need, replace=False)
            picked = picked.append(pd.Index(extra))

        return picked[:k]

    def _sample_mc(idx: pd.Index, k: int) -> pd.Index:
        """Uniform Monte Carlo sampling over indices."""
        if k <= 0:
            return pd.Index([])
        arr = idx.to_numpy()
        chosen = rng.choice(arr, size=k, replace=False)
        return pd.Index(chosen)

    # --- select subset indices ---
    if mode == 'outlier_sampling':
        selected = d_cand.nlargest(len2).index

    elif mode == 'bin_quantile_sampling':
        selected = _sample_binned(d_cand, len2)

    elif mode == 'mixture_sampling':
        # split budget
        k_bin = int(round(len2 * mixture_alpha))
        k_mc = len2 - k_bin

        # Monte Carlo part
        part_mc = _sample_mc(d_cand.index, k_mc)

        # Binned part from remaining (so we don't waste budget on duplicates)
        remaining_for_bins = d_cand.loc[d_cand.index.difference(part_mc)]
        # If MC already consumed too much and remaining is small, we’ll sample bins from full pool and dedupe/top-up
        if len(remaining_for_bins) < k_bin:
            remaining_for_bins = d_cand

        part_bin = _sample_binned(remaining_for_bins, k_bin)

        selected = part_mc.append(part_bin).unique()

        # Top-up to exactly len2 (in case of overlap)
        if len(selected) < len2:
            remaining = d_cand.index.difference(selected)
            need = len2 - len(selected)
            extra = rng.choice(remaining.to_numpy(), size=need, replace=False)
            selected = selected.append(pd.Index(extra))

        selected = selected[:len2]

    else:
        raise ValueError(
            "mode must be one of: 'outlier_sampling', 'bin_quantile_sampling', 'mixture_sampling'"
        )

    # --- build df2 ---
    df2 = test_dataframe.loc[selected]
    dists_selected = dists.loc[df2.index]

    # decode back if encoded
    if Y_cat_encoder is not None:
        df2 = Decode_all(df2, X_cat_encoder, Y_cat_encoder)
    
    

    return df2, dists_selected


def Classification_counterfactual_test_dataset(
    test_dataset: pd.DataFrame,
    model_name: str,
    X_encoders: dict,
    Y_encoders: dict,
    target_cols,
    y_excluded_values,
    n_rows: int = 10,
    model_file_path: str | None = None,
    model=None,
    target_col: str | None = None,
    random_state: int | None = None,
    compile_model: bool = True,
):
    """
    Return up to `n_rows` rows from X_test whose *predicted* value is NOT in `y_excluded_values`,
    using the generalized get_model_predictions_classification() function.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test features.
    model_name : str
        Model name, e.g. 'MLP', 'Transformer', 'Ridge', 'RandomForest', ...
    X_encoders : dict
        Encoders for X (as expected by transform_cat_encoder / prepare_NN_embedding_inputs).
    Y_encoders : dict
        Encoders for Y (as expected by postprocess_class_predictions).
    target_cols : list-like
        Target column names expected by postprocess_class_predictions.
    y_excluded_values : list / set / array-like
        List of *unencoded* target values that we want to EXCLUDE based on model prediction.
    n_rows : int, default 10
        Number of rows to return (max; if fewer are available, returns all).
    model_file : str, optional
        Path to the saved model file; used if `model` is None and for loading scalers.
    model : object, optional
        Pre-loaded model instance. If None, will be loaded from `model_file`.
    target_col : str or None, default None
        Name of the target column in the decoded predictions.
        If None, uses the first entry in `target_cols`.
    random_state : int or None, default None
        Random state for reproducible sampling.
    compile_model : bool, default True
        Passed through to Load_ModelFile for Keras models.

    Returns
    -------
    pd.DataFrame
        Subset of X_test with a column `<target_col>_pred` containing predicted values.
    """

    # 1. Get decoded predictions using the generalized helper
    
    X_test = test_dataset.drop(columns=target_cols) 
    preds = get_model_predictions_classification(
        X_test=X_test,
        model_name=model_name,
        X_encoders=X_encoders,
        Y_encoders=Y_encoders,
        target_cols=target_cols,
        model_file_path=model_file_path,
        model=model,
        compile_model=compile_model,
    )

    # 2. Figure out the target column we should use
    if target_col is None:
        if target_cols is not None and len(target_cols) > 0:
            target_col = target_cols[0]
        else:
            raise ValueError(
                "target_col not provided and `target_cols` is empty or None."
            )

    # 3. Normalize preds to a Series with the correct name
    if isinstance(preds, pd.DataFrame):
        if target_col not in preds.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in prediction columns: {list(preds.columns)}"
            )
        pred_series = preds[target_col]
    elif isinstance(preds, pd.Series):
        pred_series = preds.rename(target_col)
    else:
        # assume array-like; turn into Series
        pred_series = pd.Series(preds, index=X_test.index, name=target_col)

    # 4. Ensure excluded values is a set-like for efficient lookup
    excluded_set = set(y_excluded_values)

    # 5. Build mask: keep rows whose predicted value is NOT in excluded_set
    mask = ~pred_series.isin(excluded_set)

    test_dataset_filtered = test_dataset.loc[mask].copy()
    preds_filtered = pred_series.loc[mask]

    if test_dataset_filtered.empty:
        # nothing matches; return empty with prediction column
        test_dataset_filtered[f"{target_col}_pred"] = []
        return test_dataset_filtered

    # 6. Sample up to n_rows (if n_rows is None or >= len, just keep all)
    if n_rows is not None and n_rows < len(test_dataset_filtered):
        test_dataset_filtered = test_dataset_filtered.sample(n=n_rows, random_state=random_state)
        preds_filtered = preds_filtered.loc[test_dataset_filtered.index]

    return test_dataset_filtered, preds_filtered

def load_model_data(model, ML_files_folder):
    """Load all necessary data for the given ML model."""

    ML_files_folder = ML_files_folder
    training_data = pd.read_csv(ML_files_folder + f"{model}_train_data.csv")
    test_data = pd.read_csv(ML_files_folder + f"{model}_test_data.csv")
    meta_data = ML_files_folder + f"{model}_metadata.txt"
    if (
        model == "MLP"
        or model == "CNN_LSTM"
        or model == "CNN"
        or model == "Transformer"
    ):
        model_file = ML_files_folder + f"{model}_model.keras"
    else:
        model_file = ML_files_folder + f"{model}_model.joblib"

    return training_data, meta_data, test_data, model_file


def NN_epoch_data_editting(df: pd.DataFrame, num_epochs: int = 20, replace: bool = False) -> pd.DataFrame:
    """
    Adds a 'Cumulative_epoch_loss' column to the DataFrame by summing the first `num_epochs` epoch columns. 
    Has the option of removing epoch columns for simplicity 

    Parameters:
        df (pd.DataFrame): The input dataframe.
        num_epochs (int): Number of epoch columns to include in the sum.
        replace (bool): If True, removes the epoch columns used in the sum.

    Returns:
        pd.DataFrame: Modified dataframe with the new column added.
    """
    # Identify and sort epoch columns numerically
    epoch_cols = sorted(
        [col for col in df.columns if col.startswith('epoch ')],
        key=lambda x: int(x.split()[1])
    )

    # Select the first `num_epochs`
    selected_epoch_cols = epoch_cols[:num_epochs]

    # Add new column with row-wise sum
    df['Cumulative_epoch_loss'] = df[selected_epoch_cols].sum(axis=1)

    # Drop selected epoch columns if requested
    if replace:
        df = df.drop(columns=selected_epoch_cols)

    return df

def split_test_data(test_data,X_labels,Y_labels):
    """Split test data into X and Y components."""
    test_data_X = test_data[X_labels]
    test_data_Y = test_data[Y_labels]
    return test_data_X, test_data_Y


# for i in range(len(test_data_X)):
#     #Create a string of the test datapoint
#     

def Test_data_point_extractor(test_data_X_row,test_data_Y_row):
    '''Extracts test datpoint for the prompt to the LLM
    
   Input:
       test_data_X_row: value of a certain test_data_X row
       test_data_Y_row: value of corresponding test_data_y row
        
    
    Output:
        Test_Data_X and Test_data_Y output as strings where Column:value
    
    '''
    Test_datapoint_X_string = ", ".join([f"{col}: {val}" for col, val in zip(test_data_X_row.index, test_data_X_row)])
    Test_datapoint_Y_string=", ".join([f"{col}: {val}" for col, val in zip(test_data_Y_row.index, test_data_Y_row)])
 
    return Test_datapoint_X_string,Test_datapoint_Y_string

def clean_code_block(result: str) -> str:
    """
    Removes Markdown-style code block formatting (e.g., ```python ... ```)
    from a given string.

    Args:
        result (str): The raw LLM output string.

    Returns:
        str: Cleaned string without triple backticks or language specifiers.
    """
    cleaned_result = result.strip()
    if cleaned_result.startswith("```"):
        cleaned_result = "\n".join(
            line for line in cleaned_result.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    return cleaned_result

def LLM_answers(LLMs,role,content):
    '''fetches answers from APIs of LLMs for a given prompt
    
   Input:
       LLMs:List of APIs to be called, which are in LLM_API_calls.py
       role: role field of LLM arguments
       content: content field of LLM arguments
       
    
    Output:
        dictionary of different LLM's answers for a particular prompt (test datapoint)'
    
    '''
    LLM_predictions = {llm: [] for llm in LLMs}
    for llm_name in LLMs:
            
            #First checking if it is a together model
            result=Together_models(role, content, llm_name)

            if result==False:
                LLM_API = getattr(LLM_API_calls, llm_name)
                result=LLM_API(role,content)

            cleaned_result = clean_code_block(result)
            LLM_predictions[llm_name].append(cleaned_result)
   
    return LLM_predictions

        
        
        
def Prompt_creator_Clusterer(Prompt_generator_parameters, test_datapoint_X, test_datapoint_Y):
    '''Create a prompt based for a particular test datapoint for 
    a given Machine learning model to be explained by an LLM
    
   Input:
       test_datapoint_X: X variable value for the test datapoint
       test_datapoint_Y: X variable value for the test datapoint
       
       Prompt_generator_parameters Dictionary:
           metric=metric of evaluation Accuracy, counterfactuals etc,
           model=model file name
           meta_data=meta_data_filename
           training_data=training data dataframe
           output_variable_number=number of regression outputs  
    
    Output:
        prompt as 2 variables role,content
    
    '''
    #Arguments to be used in prompt creation
    args=Prompt_generator_parameters
    
    pobj=Prompt_generator(args)
    #Create prompt
    role=pobj.LLM_role()
    Question=pobj.Question(test_datapoint_X, test_datapoint_Y, args["output_upgrade_CF"])
    
    
    Clusterer_output=pobj.TD_sample_data_Clusterer( test_datapoint_X, test_datapoint_Y, 
                                                   args["output_upgrade_CF"], args['output_variable_number'],
                                                   Cluster_class_obj=args["Cluster_classifier_object"])
        
    
    content = (
        pobj.Dataset_info() +
        Question+
        pobj.Tasks_string(args['output_variable_number'])+
        pobj.JG_string()+
        pobj.Output_format(args['output_variable_number'])+
        pobj.Constraints_string()+
        Clusterer_output
        )

    return role,content, Question+Clusterer_output

def Baseline_prediction_function(LLMs, args, test_datapoint_X, random_sample_length=50):
    
    training_data=args['training_data']
    X_cat_encoder=args["X_cat_encoder"]
    Y_cat_encoder=args["Y_cat_encoder"]
    
    
    #Encoding Training dataframe for Classification tasks
    if (Y_cat_encoder!=None):
        
        #encoding Training dataframe
        df_unencoded=training_data.copy()
        df_encoded=transform_cat_encoder(df_unencoded, X_cat_encoder)
        training_data=df_encoded



    Baseline_predictions = {llm: [] for llm in LLMs}
    for llm_name in LLMs:
        model_file_path = args["ML_model_file_path"]
        model_name=args["ML_model_name"]

        # Load scalers only for models that were trained with scaled inputs
        
        if args["metric"] == 'Accuracy':
            # sample background in the SAME column order as Inputs
            
            X_background = training_data[args['Inputs']].sample(n=random_sample_length, random_state=random_sample_length).reset_index(drop=True)

            preds, local_feature_attribution = Lime_wrapper_function(
                model_file_path=model_file_path,
                model_name=model_name,
                x_row=test_datapoint_X,   # string your parser understands
                feature_names=args['Inputs'], 
                Target_cols=args['Targets'],
                X_background=X_background,    # DataFrame
                X_cat_encoder=X_cat_encoder,
                Y_cat_encoder=Y_cat_encoder,
                y_index_for_lime=0
            )
            result = format_predictions_and_importances_separate(preds, local_feature_attribution, 'Accuracy')
            string_result=str(result)
            Baseline_predictions[llm_name].append(string_result) 

        elif args["metric"] == 'Counterfactuals':
            # sample background in the SAME column order as Inputs
            X_background = training_data[args['Inputs']].sample(n=random_sample_length, random_state=random_sample_length).reset_index(drop=True)
            Immutables=args['Immutables']
            preds = Dice_wrapper_function(
                model_file_path=model_file_path,
                model_name=model_name,
                x_row=test_datapoint_X,  
                Input_features=args['Inputs'],
                Target_cols=args['Targets'],
                immutable_features=Immutables,
                output_upgrade_list=args["output_upgrade_CF"],                    # list -> used directly
                feature_delta_grid=args["Dice_grid"],
                X_background=X_background,
                X_cat_encoder=X_cat_encoder,
                Y_cat_encoder=Y_cat_encoder,
                max_combos=50)
            
            #blank feature attributes as Dice doesnt give FA
            local_feature_attribution={target: {feature: 0.0 for feature in args['Inputs']}for target in args['Targets']}
            
            result = format_predictions_and_importances_separate(preds, local_feature_attribution, 'Counterfactuals')
            string_result=str(result)
            Baseline_predictions[llm_name].append(string_result)
        
        elif args["metric"] == 'Feature_importance':
            random_sample_length=50
            attributes_all_outputs=[]
            # sample background in the SAME column order as Inputs
            training_data=args['training_data']
            X_background = training_data[args['Inputs']].sample(n=random_sample_length, random_state=random_sample_length).reset_index(drop=True)

            for i in range (args["output_variable_number"]):
                attributes_feature = Shap_feature_attributions(model_name=args["ML_model_name"] ,
                                                     model=model_name,
                                                     x_row=test_datapoint_X,
                                                     X_background=X_background,
                                                     y_index=i)
                
                attributes_feature=[abs(num) for num in attributes_feature]
                attributes_all_outputs.append([round(num, 2) for num in attributes_feature]) 
        
            string_result=str(attributes_all_outputs)
            Baseline_predictions[llm_name].append(string_result)
        
    
            
        
    return Baseline_predictions
            
    


def Evaluation_all_answers(LLM,Evaluation_parameters,increment=None):
     """Evaluate LLM prdedictions wrt ground truth or XAI predictions based on 
     a particular metric (Accuracy, Counterfactuals etc...)
     
     Inputs:
         LLM: name of LLM used for prediction
         Evaluation parameters (dictionary):
              
              "metric": metric of evaluation,
              "LLM_predictions":dictionary containing LLM_predictions for which we will evaluate,
              "meta_data_filename": meta_data_filename,
              "training_data": training_data,
              "test_data_X": test_data_X,
              "test_data_Y": test_data_Y,
              "output_variable_number": output_variables,
              "model": model_file,
              "model_name": model_name,
              "X_labels":X_labels,
              "Y_labels":Y_labels
         
         increment: array for counterfactuals output upgrade
     
     Output:
         l1_norm: l1 norm
         l2 norm: l2 norm
         LLM_df: LLM's answers
         ML_df: ML model's answers'
        '''
     
     """
     #Creating object of evaluation class
     Evaluation_obj=Evaluation_metrics(LLM, Evaluation_parameters)  
     # Separately calling methods 'Counterfactuals' and 'Accuracy' due to different parameter list      
     
     if (Evaluation_parameters['ML_task_type']=='regression'):
         if(Evaluation_parameters['metric']=='Counterfactuals'):
             metrics_report, LLM_df, ML_df, LLM_raw_answers, dropped_test_answers=Evaluation_obj.Counterfactuals(output_upgrade=increment, Classification_problem=False)
             
         elif(Evaluation_parameters['metric']=='Accuracy'):
             metrics_report, LLM_df, ML_df, LLM_raw_answers, dropped_test_answers=Evaluation_obj.Accuracy()
     
     elif (Evaluation_parameters['ML_task_type']=='classification'):
         if(Evaluation_parameters['metric']=='Counterfactuals'):
             metrics_report, LLM_df, ML_df, LLM_raw_answers, dropped_test_answers=Evaluation_obj.Counterfactuals(output_upgrade=increment, Classification_problem=True)
             
         elif(Evaluation_parameters['metric']=='Accuracy'):
             metrics_report, LLM_df, ML_df, LLM_raw_answers, dropped_test_answers=Evaluation_obj.Accuracy(Classification_problem=True)
     
     
     return metrics_report, LLM_df, ML_df, LLM_raw_answers, dropped_test_answers
 

def extract_subset_dict(d, keys):
    return {k: d[k] for k in keys if k in d}


def format_predictions_and_importances_separate(
    preds_dict: Dict[str, float],
    weights_dict: Dict[str, Dict[str, float]],
    metric:str,
    round_preds: int = 2,
    round_importances: int = 4) -> Dict[str, Dict]:
    """
    Combine predictions and per-target feature importances.
    Assumes:
      preds_dict  = { "<target>_predicted": float, ... }
      weights_dict = { "<target>_predicted": { feature: weight, ...}, ... }

    Rounds predictions and feature importances.
    """

    if metric=='Accuracy':
        label='prediction'
    elif metric=='Counterfactuals':
        label='counterfactual_input'

    # strip intercept if present & round importances
    feature_importances = {
        t: {fname: round(float(val), round_importances)
            for fname, val in wdict.items() if fname != "intercept"}
        for t, wdict in weights_dict.items()
    }

    def safe_round(v, ndigits):
        if isinstance(v, (float, int, np.number)):
            return round(float(v), ndigits)
        return v  # string labels, etc.
    
    return {
        label: {k: safe_round(v, round_preds) for k, v in preds_dict.items()},
        "feature_importances": feature_importances
    }

def format_llm_answers_with_labels(answer_dict: dict, X_labels: list) -> str:
    """
    Formats LLM answers into a single labeled string with LLM names included.

    Args:
        answer_dict (dict): Dictionary with LLM names as keys and list of answer strings as values.
        X_labels (list): List of feature names corresponding to the values.

    Returns:
        str: Combined formatted string from all LLMs, wrapped with line breaks.
    """
    final_output = "\n\n"

    for llm_name, answer_list in answer_dict.items():
        try:
            # Remove code block markers
            raw_answer = answer_list[0].strip().replace("```python", "").replace("```", "")
            parsed_list = ast.literal_eval(raw_answer)

            # Combine with labels
            labeled_string = ", ".join(f"{label}: {value}" for label, value in zip(X_labels, parsed_list))
            final_output += f"{llm_name}: {labeled_string}\n"

        except Exception as e:
            final_output += f"{llm_name}: [Error parsing answer: {e}]\n"

    final_output += "\n\n"
    return final_output

def Temp_ground_truth_to_string(ground_truth: dict, output_keys: list, LLMs) -> str:
    """
    Convert a ground truth dictionary into a feature_importances string.
    
    Args:
        ground_truth (dict): Dictionary of the form {
            'target_0': {'RSRQ': ..., 'RSSI': ..., ...},
            'target_1': {...}, ...
        }
        output_keys (list): List of output labels to assign, e.g.
            ['DL_bitrate_predicted', 'RSRP_predicted', 'SNR_predicted']
    
    Returns:
        str: Dictionary-like string formatted as 'feature_importances': {...}
    """
    GT_predictions={llm: [] for llm in LLMs}
    for llm_name in LLMs:
        targets = sorted(ground_truth.keys(), key=lambda k: int(k.split("_")[-1]))
        if len(targets) != len(output_keys):
            raise ValueError("Number of targets and output_keys must match.")
        
        # remap targets → output_keys
        remapped = {ok: ground_truth[t] for ok, t in zip(output_keys, targets)}
        
        # pretty JSON string
        dict_str = f"'feature_importances': {remapped}"
            
    
    
    
    GT_predictions[llm_name].append(dict_str)
    
    return GT_predictions
    