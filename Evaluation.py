# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:12:16 2025

@author: arnab
"""

import pandas as pd
import numpy as np
import ast
import math
import joblib
from typing import Sequence, Optional, Dict, Any, List, Tuple, Iterable
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


from Trained_ML_model_functions import Load_ModelFile,  get_model_predictions_classification, get_model_predictions_regression
from Cat_variable_functions import ( 
                                    transform_cat_encoder, prepare_NN_embedding_inputs, 
                                    postprocess_class_predictions, 
                                    Ridge_prepare_inputs_from_df)



class Evaluation_metrics:
    
    '''
    initialize class with the following parameters and call the evaluation function
    
    Initializing variables:
        LLM: The LLM for which we are evaluating
        Evaluation dictionary:
             
             "metric": metric of evaluation,
             "LLM_predictions":dictionary containing LLM_predictions for which we will evaluate,
             "LLM_models": LLMs used,
             "test_data_X": test_data_X,
             "test_data_Y": test_data_Y,
             "output_variable_number": output_variables,
             "model": model_file,
             "model_name": model_name,
    
   '''
    def __init__(self,LLM_model,Evaluation_parameters):
        
        self.LLM_result_dict=Evaluation_parameters["LLM_predictions"][LLM_model]
        self.Use_case_training_data=Evaluation_parameters['ML_training_dataframe']
        self.Test_data_X=Evaluation_parameters["test_data_X"]
        self.Test_data_Y=Evaluation_parameters["test_data_Y"]
        self.model_file=Evaluation_parameters["model"]
        self.model_name=Evaluation_parameters["model_name"]
        self.X_labels=Evaluation_parameters["X_labels"]
        self.Y_labels=Evaluation_parameters["Y_labels"]
        self.Immutables=Evaluation_parameters["Immutables"]
        self.target_cols=Evaluation_parameters["Targets"]
        self.output_variable_number=Evaluation_parameters["output_variable_number"]
        self.ML_task_type=Evaluation_parameters["ML_task_type"]
        self.X_encoders=Evaluation_parameters['classification_X_encoder']
        self.Y_encoders=Evaluation_parameters['classification_Y_encoder']
   
#---------------Callable functions-----------------------------

        
    
    def Counterfactuals(self, output_upgrade, Classification_problem:bool =False,):
        rows, kept_idx, dropped_idx = self._preprocessing(self.LLM_result_dict, 'Counterfactuals')

        self.discarded_counterfactual_indices = dropped_idx
        self.kept_counterfactual_indices = kept_idx

        if len(rows) == 0:
            print("No valid LLM rows after preprocessing for Counterfactuals.")
            # Return empty artifacts to avoid crashing downstream
            empty_df = pd.DataFrame(columns=self.X_labels)
            empty_y = self.Test_data_Y.iloc[[]]
            return {}, np.empty((0, getattr(self, 'output_variable_number', 0))), empty_y, empty_df, dropped_idx

        # Build LLM_raw_df from list-of-dicts using X_labels as the column order
        try:
            if Classification_problem==False:
                data_matrix = [[float(row[k]) for k in self.X_labels] for row in rows]
            else:
                data_matrix = [[row[k] for k in self.X_labels] for row in rows]
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(
                f"Counterfactuals row missing expected key '{missing}'. "
                f"Each dict must have keys exactly matching X_labels: {list(self.X_labels)}"
            ) from e
            
        

        LLM_raw_df = pd.DataFrame(data_matrix, columns=self.X_labels)
        # To guard against NaNs:
        
        if Classification_problem==False:
            LLM_raw_df = LLM_raw_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        

        # Positional alignment for ground truth
        X_test_aligned = self.Test_data_X.iloc[kept_idx].reset_index(drop=True)  # kept for symmetry/debug
        Y_test_aligned = self.Test_data_Y.iloc[kept_idx].reset_index(drop=True)
        
        if Classification_problem==True:
            LLM_upgraded_predictions =  get_model_predictions_classification(X_test=LLM_raw_df,
                                                                    model_name=self.model_name,
                                                                    X_encoders=self.X_encoders,
                                                                    Y_encoders=self.Y_encoders,
                                                                    target_cols=self.target_cols,
                                                                    model_file_path=self.model_file,   # model will be loaded inside
                                                                    model=None)              # or pass a pre-loaded model instead
                                                                
    
        else:
            LLM_upgraded_predictions = get_model_predictions_regression(X_test=LLM_raw_df,
                                                              model_name=self.model_name,
                                                              model_file_path=self.model_file)
        
        LLM_upgraded_predictions = np.asarray(LLM_upgraded_predictions)
        
        # Ensure 2D shape
        if LLM_upgraded_predictions.ndim == 1:
            LLM_upgraded_predictions = LLM_upgraded_predictions.reshape(-1, 1)

        if Classification_problem==False:
            # Build expected performance = ground truth + output_upgrade (vector)
            Expected_performance = Y_test_aligned.apply(pd.to_numeric, errors='coerce').copy()

            if len(output_upgrade) != Expected_performance.shape[1]:
                raise ValueError(
                    f"Length of output_upgrade ({len(output_upgrade)}) must match number of outputs "
                    f"({Expected_performance.shape[1]})."
                )

            Expected_performance = Expected_performance.add(pd.Series(output_upgrade, index=Expected_performance.columns), axis=1)
            
        elif Classification_problem==True:
            # Number of rows
            n_rows = LLM_upgraded_predictions.shape[0]

            # Repeat output_upgrade for every row
            data = np.tile(output_upgrade, (n_rows, 1))  # shape (n_rows, len(output_upgrade))

            Expected_performance = pd.DataFrame(data, columns=self.Y_labels)


        
        if Classification_problem==False:
            expected_np = Expected_performance.to_numpy(dtype=float)
            pred_np = np.asarray(LLM_upgraded_predictions, dtype=float)
            
            if expected_np.shape != pred_np.shape:
                raise ValueError(f"Shape mismatch: expected {expected_np.shape}, got {pred_np.shape}")

            
            metrics_report=cf_metrics(expected_np, pred_np, self.Test_data_X, LLM_raw_df, self.target_cols, self.Immutables, self.Use_case_training_data, ML_task_type='regression')
        else:
            expected_np = Expected_performance.to_numpy()
            pred_np = np.asarray(LLM_upgraded_predictions)
            
            if expected_np.shape != pred_np.shape:
                raise ValueError(f"Shape mismatch: expected {expected_np.shape}, got {pred_np.shape}")
            
            metrics_report=cf_metrics(expected_np, pred_np, self.Test_data_X, LLM_raw_df, self.target_cols, self.Immutables, self.Use_case_training_data, ML_task_type='classification')
            


        return (
            metrics_report,
            LLM_upgraded_predictions,
            Expected_performance,
            LLM_raw_df,
            dropped_idx
        )
     
    
    

    def Accuracy(self, Classification_problem:bool =False, version='MAE'):
        rows, kept_idx, dropped_idx = self._preprocessing(self.LLM_result_dict, 'Accuracy')

        if len(rows) == 0:
            print("No valid LLM rows after preprocessing for Accuracy.")
            # Return empty artifacts to avoid crashing downstream
            return {}, np.empty((0, getattr(self, 'output_variable_number', 0))), np.empty((0, getattr(self, 'output_variable_number', 0))), None, dropped_idx

        # Convert list-of-dicts -> 2D array using Y_labels as the column order
        try:
            if Classification_problem==True:
                
                labels=[f"{k}_predicted" for k in self.Y_labels]
                LLM_predictions_np = np.array([[row[k] for k in labels] for row in rows])
           
            else:
                labels=[f"{k}_predicted" for k in self.Y_labels]
                LLM_predictions_np = np.array([[float(row[k]) for k in labels] for row in rows], dtype=float)
            
            
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(f"LLM row missing expected key '{missing}'. "
                             f"Make sure each dict has keys exactly matching Y_labels: {list(self.Y_labels)}") from e
       

        # Positional selection, not label drop
        X_test_aligned = self.Test_data_X.iloc[kept_idx].reset_index(drop=True)
        

        if Classification_problem==True:
            ML_predictions =  get_model_predictions_classification(X_test=X_test_aligned,
                                                                    model_name=self.model_name,
                                                                    X_encoders=self.X_encoders,
                                                                    Y_encoders=self.Y_encoders,
                                                                    target_cols=self.target_cols,
                                                                    model_file_path=self.model_file,   # model will be loaded inside
                                                                    model=None)              # or pass a pre-loaded model instead
                                                                
    
        else:
            ML_predictions = get_model_predictions_regression(X_test=X_test_aligned,
                                                              model_name=self.model_name,
                                                              model_file_path=self.model_file)
                                                          
  
        if Classification_problem==False:
            ML_predictions = np.asarray(ML_predictions, dtype=float).reshape(-1, self.output_variable_number)
        else:
            ML_predictions = np.asarray(ML_predictions).reshape(-1, self.output_variable_number)


            

        if LLM_predictions_np.shape[0] != ML_predictions.shape[0]:
            raise ValueError(f"Row mismatch after alignment: LLM={LLM_predictions_np.shape[0]} vs ML={ML_predictions.shape[0]}")

        
        if Classification_problem==True:
            metrics_report = compute_multioutput_metrics_classification(ML_predictions, LLM_predictions_np)
        else:
            metrics_report = compute_multioutput_metrics_regression(ML_predictions, LLM_predictions_np)

        LLM_raw_df = None
        return metrics_report, LLM_predictions_np, ML_predictions, LLM_raw_df, dropped_idx
    
    
    
    def Feature_importance(self):
        #Ground_truth_answer_FA=F_importance_ground_truth_wrapper(model_name, model_file, x_row, X_labels, num_outputs)
        pass
    

 #---------------Internal-----------------------------   
    
    def _preprocessing(self, LLM_results, metric, CF_test_X=None):
        """
        Filter, validate, and (optionally) repair LLM JSON outputs for:
          - 'Accuracy'       (target-side, e.g. {..., "LOS_CLASS_predicted": ...})
          - 'Counterfactuals' (feature-side, e.g. {"AGE": ..., "GENDER": ..., ...})

        Behaviour
        ---------
        1) Ensures each LLM result row:
           - is a non-empty dict
           - has exactly the expected keys (by count and name)
        2) For classification:
           - Validates each value against the corresponding LabelEncoder:
             * either as encoded integer (0..n_classes-1)
             * or as a raw label in encoder.classes_
        3) For classification + Counterfactuals + immutable features:
           - If a value is invalid AND the column is in self.Immutables,
             we try to "repair" it by replacing with the original value from:
               - CF_test_X (if given), else
               - self.Test_data_X
             If that original value is valid for the encoder, we keep the row.
             Otherwise, the row is dropped.

        Parameters
        ----------
        LLM_results : list[dict]
            Raw LLM JSON outputs (one per test row).
        metric : str
            Either 'Accuracy' or 'Counterfactuals'.
        CF_test_X : optional
            Feature matrix corresponding to the same rows as LLM_results.
            Used for "repairing" immutable features if values are invalid.
            If None, self.Test_data_X is used.

        Returns
        -------
        kept_rows : list[dict]
            The (possibly repaired) LLM result rows that passed all checks.
        kept_idx : list[int]
            Indices into LLM_results that were kept.
        dropped_idx : list[int]
            Indices into LLM_results that were dropped (any reason).
        """

        # --- Determine expected structure (keys & lengths) ---
        if metric == "Accuracy":
            # e.g. self.Y_labels = ['LOS_CLASS']
            req_len = len(self.Y_labels)
            expected_keys = {f"{k}_predicted" for k in self.Y_labels}

        elif metric == "Counterfactuals":
            # e.g. self.X_labels = ['AGE', 'GENDER', ...]
            req_len = len(self.X_labels)
            expected_keys = set(self.X_labels)

        else:
            raise ValueError("Invalid metric. Choose 'Accuracy' or 'Counterfactuals'.")

        # --- Build encoder mapping by column name (for classification only) ---
        if self.ML_task_type == "regression":
            encoders = {}
        elif self.ML_task_type == "classification":
            if metric == "Counterfactuals":
                # Feature encoders: only for columns that actually have encoders
                encoders = {k: self.X_encoders[k] for k in self.X_labels if k in self.X_encoders}
            elif metric == "Accuracy":
                # Target encoders: often stored as a dict or single encoder
                # Map each "<Y>_predicted" key to the same encoder bundle
                encoders = {f"{k}_predicted": self.Y_encoders for k in self.Y_labels}
        else:
            raise ValueError(f"Unknown ML_task_type: {self.ML_task_type}")

        # --- Output bookkeeping ---
        kept_rows = []
        kept_idx = []
        dropped_idx = []
        bad_value_idx = []   # indices dropped due to encoder incompatibility
        repaired_idx = []    # indices where immutable value was repaired

        # --- Helper: check if 'value' is valid for encoder_for_col ---
        def value_valid_for_encoder(value, encoder_for_col):
            if isinstance(encoder_for_col, dict):
                if len(encoder_for_col) != 1:
                    raise ValueError(f"Encoder dict has {len(encoder_for_col)} entries, expected 1.")
                le = next(iter(encoder_for_col.values()))
            else:
                le = encoder_for_col

            classes = le.classes_
            classes_are_numeric = np.issubdtype(np.array(classes).dtype, np.number)

            # int provided by LLM
            if isinstance(value, (int, np.integer)):
                # Only accept ints if the *labels themselves* are numeric
                return (value in classes) if classes_are_numeric else False

            # numeric string
            if isinstance(value, str) and value.isdigit():
                iv = int(value)
                return (iv in classes) if classes_are_numeric else False

            return value in classes
    

        # --- Helper: attempt to repair an immutable feature value ---
        def try_repair_immutable(row_idx, col_name, encoder_for_col, bad_val):
            """
            Attempt to repair immutable feature 'col_name' in row 'row_idx'
            by replacing the LLM-provided 'bad_val' with the original
            value from CF_test_X (if not None) or self.Test_data_X.

            Returns
            -------
            (success, repaired_value)
            """
            src_X = CF_test_X if CF_test_X is not None else self.Test_data_X

            if src_X is None:
                print(f"[Repair] No source data available to repair immutable '{col_name}' for row {row_idx}.")
                return False, bad_val

            try:
                # DataFrame path
                if hasattr(src_X, "iloc") and hasattr(src_X, "columns"):
                    if col_name not in src_X.columns:
                        print(
                            f"[Repair] Column '{col_name}' not in source DataFrame "
                            f"for row {row_idx}."
                        )
                        return False, bad_val
                    original_val = src_X.iloc[row_idx][col_name]
                else:
                    # Assume numpy array with columns matching self.X_labels
                    col_idx = self.X_labels.index(col_name)
                    original_val = src_X[row_idx, col_idx]

                if value_valid_for_encoder(original_val, encoder_for_col):
                    print(
                        f"[Repair] Immutable '{col_name}' at row {row_idx}: "
                        f"LLM value '{bad_val}' -> original '{original_val}'."
                    )
                    return True, original_val
                else:
                    print(
                        f"[Repair] Immutable '{col_name}' at row {row_idx} has "
                        f"invalid original value '{original_val}' for encoder as well."
                    )
                    return False, bad_val

            except Exception as e:
                print(
                    f"[Repair] Failed to repair immutable '{col_name}' at row {row_idx}: {e}"
                )
                return False, bad_val

        # --- Main loop over LLM results ---
        for i, item in enumerate(LLM_results):
            try:
                # 1) Must be a non-empty dict
                if not item or not isinstance(item, dict):
                    dropped_idx.append(i)
                    continue

                # 2) Wrong number of keys
                if len(item) != req_len:
                    dropped_idx.append(i)
                    continue

                # 3) Wrong key names
                if set(item.keys()) != expected_keys:
                    dropped_idx.append(i)
                    continue

                # 4) Classification-specific: validate values using encoders
                if self.ML_task_type == "classification":
                    valid_row = True
                    row_repaired = False

                    for col, val in item.items():
                        # Only validate columns that have encoders
                        if col in encoders:
                            enc_for_col = encoders[col]

                            if not value_valid_for_encoder(val, enc_for_col):

                                # --- Try repair for immutable feature (Counterfactuals only) ---
                                repaired = False
                                if (
                                    metric == "Counterfactuals"
                                    and hasattr(self, "Immutables")
                                    and col in getattr(self, "Immutables", [])
                                ):
                                    repaired, new_val = try_repair_immutable(
                                        row_idx=i,
                                        col_name=col,
                                        encoder_for_col=enc_for_col,
                                        bad_val=val,
                                    )
                                    if repaired:
                                        item[col] = new_val
                                        row_repaired = True

                                # If still invalid after repair attempt, drop the row
                                if not repaired:
                                    valid_row = False
                                    print(f"Problem column: {col} , LLM provided value : {val}")
                                    break  # stop checking remaining columns for this row

                    if not valid_row:
                        # Dropped due to encoder incompatibility
                        bad_value_idx.append(i)
                        dropped_idx.append(i)
                        continue

                    if row_repaired:
                        repaired_idx.append(i)

                # If all checks passed (and any repairs succeeded), keep the row
                kept_rows.append(item)
                kept_idx.append(i)

            except Exception as e:
                # Any unexpected error -> drop the row
                print(f"[Exception] Dropping row {i} due to error: {e}")
                dropped_idx.append(i)

        # --- Logging ---
        print(f"\nTotal rows: {len(LLM_results)}")
        print(f"Kept rows: {len(kept_rows)}")
        print(f"Dropped rows: {len(dropped_idx)}")

        if self.ML_task_type == "classification":
            print(f"• Dropped due to invalid values (encoder mismatch): {len(bad_value_idx)}")
            print(f"  Indices: {bad_value_idx}")
            print(f"• Repaired immutable-value rows: {len(repaired_idx)}")
            print(f"  Indices: {repaired_idx}\n")

        return kept_rows, kept_idx, dropped_idx
    

    def _preprocessing_old(self, LLM_results, metric, CF_test_X=None):
        """
        Filter and validate LLM JSON outputs for:
          - 'Accuracy'   (targets)
          - 'Counterfactuals' (features)

        Keeps only rows that:
          - are valid dicts
          - have exactly the required keys
          - (for classification) have values compatible with the LabelEncoder(s).

        Returns
        -------
        kept_rows : list[dict]
        kept_idx  : list[int]
        dropped_idx : list[int]
        """

        # --- Determine expected structure ---
        if metric == 'Accuracy':
            # e.g. Y_labels = ['LOS_CLASS']
            req_len = len(self.Y_labels)
            expected_keys = {f"{k}_predicted" for k in self.Y_labels}
    
            

        elif metric == 'Counterfactuals':
            # e.g. X_labels = ['AGE', 'GENDER', ...]
            req_len = len(self.X_labels)
            expected_keys = set(self.X_labels)
        
        else:
            raise ValueError("Invalid metric. Choose 'Accuracy' or 'Counterfactuals'.")
        
            
        if self.ML_task_type == 'regression':
            encoders = {}
        elif self.ML_task_type == 'classification':
            if metric == 'Counterfactuals':
                encoders = {k: self.X_encoders[k] for k in self.X_labels if k in self.X_encoders}
            elif metric == 'Accuracy':
                encoders = {f"{k}_predicted": self.Y_encoders for k in self.Y_labels}
        
        # --- Output bookkeeping ---
        kept_rows = []
        kept_idx = []
        dropped_idx = []
        bad_value_idx = []   # specifically for invalid encoder values

        # --- Helper: check if 'value' is valid for encoder_for_col ---
        def value_valid_for_encoder(value, encoder_for_col):
            """
            encoder_for_col can be:
              - a LabelEncoder
              - a dict like {"LOS_CLASS": LabelEncoder()}

            We just:
              1) get the underlying LabelEncoder
              2) check if 'value' is either:
                 - a valid encoded int (0..n_classes-1), or
                 - a valid raw label in encoder.classes_
            """
            # 1) unwrap to a single LabelEncoder
            if isinstance(encoder_for_col, dict):
                # e.g. {"LOS_CLASS": LabelEncoder()}
                if len(encoder_for_col) != 1:
                    raise ValueError(f"Encoder dict has {len(encoder_for_col)} entries, expected 1: {encoder_for_col}")
                le = next(iter(encoder_for_col.values()))
            else:
                # assume it's already a LabelEncoder
                le = encoder_for_col

            # 2) integer-encoded (or np.integer)
            if isinstance(value, (int, np.integer)):
                return 0 <= value < len(le.classes_)

            # 3) numeric string (e.g. "0", "1") -> treat like encoded int
            if isinstance(value, str) and value.isdigit():
                iv = int(value)
                return 0 <= iv < len(le.classes_)

            # 4) otherwise, check as raw label
            return value in le.classes_
        

        # --- Main loop over LLM results ---
        for i, item in enumerate(LLM_results):
            try:
                # 1) Must be a non-empty dict
                if not item or not isinstance(item, dict):
                    dropped_idx.append(i)
                    print('empty dict')
                    continue

                # 2) Wrong number of keys
                if len(item) != req_len:
                    dropped_idx.append(i)
                    print(f'Lesser items in X_input: {len(item)} !={len(item)}')
                    continue

                # 3) Wrong key names
                if set(item.keys()) != expected_keys:
                    dropped_idx.append(i)
                    print(f'key error:{item.keys()} !={expected_keys}')
                    continue

                # 4) Classification-specific: validate values using encoders
                if self.ML_task_type == 'classification':
                    valid_row = True
                    for col, val in item.items():
                        # We only validate cols that have encoders
                        if col in encoders:
                            enc_for_col = encoders[col]
                            if not value_valid_for_encoder(val, enc_for_col):
                                valid_row = False
                                print(f'Problem column: {col} , LLM provided value : {val}')
                                break

                    if not valid_row:
                        print('not a valid row block')
                        bad_value_idx.append(i)
                        dropped_idx.append(i)
                        continue

                # If all checks passed, keep the row
                kept_rows.append(item)
                kept_idx.append(i)

            except Exception:
                # Any unexpected error -> drop the row
                print('removed row from exception block due to unexpected error')
                dropped_idx.append(i)

        # --- Logging ---
        print(f"\nTotal dropped: {len(dropped_idx)} rows")
        if self.ML_task_type == 'classification':
            print(f"• Dropped due to invalid values (not matching encoder): {len(bad_value_idx)} rows")
            print(f"• Their indices: {bad_value_idx}\n")

        return kept_rows, kept_idx, dropped_idx
 
    def extract_nth(data, n):
        """
        data : list of lists with string inside
        n    : index of sublist you want (0 for first, 1 for second, etc.)
        """
        result = []
        for inner in data:
            # each `inner` is a list with 1 string
            list_str = inner[0]  
            parsed = ast.literal_eval(list_str)  # convert string to Python list
            result.append(parsed[n])  # take nth sublist
        return result
    
    
    
    
#----------------Global Helper functions---------
    

def Single_point_score(model_file, model_input, LLM_answer):
    #import model to see model's predictions for test data X
    model = Load_ModelFile(model_file)
    
    #Reshape model_inputs and convert LLM answers to an array of floats
    model_input=model_input.values.reshape(1, -1)
    LLM_answer_list = ast.literal_eval(LLM_answer[0]) 
    LLM_answer_float = np.array(LLM_answer_list, dtype=float)
    
    predictions = model.predict(model_input).flatten()
    score = round(np.abs( LLM_answer_float  - predictions).sum(),2)
    
    #Check for Nans and give a disclaimer
    if(math.isnan(score)):
        print('LLM answer is NaN')
        #replacing NaNs with zero
        score = 0
    
    
    
    return score



#for CNN_LSTM only
def make_windows(X, time_steps=3):
    import numpy as np
    N = len(X)
    if N < time_steps:
        raise ValueError("Not enough rows for chosen time_steps.")
    return np.stack([X[i:i+time_steps] for i in range(N - time_steps + 1)], axis=0)

def compute_multioutput_metrics_classification(
    y_true: np.ndarray,          # shape (N, T)
    y_pred: np.ndarray,          # shape (N, T)
    target_names: Optional[Sequence[str]] = None,
    round_digits: Optional[int] = 3) -> Dict[str, Any]:
    """
    Multi-output classification metrics.

    Returns:
      {
        "per_target": pd.DataFrame (rows = targets, cols = metrics),
        "overall": dict of macro/micro metrics,
        "confusion_matrices": dict[target_name] -> np.ndarray,
        "n_samples": int
      }
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    N, T = y_true.shape

    if target_names is None:
        target_names = [f"y{i}" for i in range(T)]
    if len(target_names) != T:
        raise ValueError("len(target_names) must equal number of targets (T)")

    # Per-target metrics
    acc_list = []

    macro_f1_list = []
    micro_f1_list = []
    weighted_f1_list = []

    macro_prec_list = []
    micro_prec_list = []
    weighted_prec_list = []

    macro_rec_list = []
    micro_rec_list = []
    weighted_rec_list = []

    n_classes_list = []
    valid_n_list = []

    confusion_matrices: Dict[str, np.ndarray] = {}

    for j in range(T):
        yj = y_true[:, j]
        pj = y_pred[:, j]

        # handle possible NaNs (if labels are floats)
        if np.issubdtype(yj.dtype, np.floating) or np.issubdtype(pj.dtype, np.floating):
            mask = np.isfinite(yj) & np.isfinite(pj)
        else:
            # assume no NaN for integer/str labels
            mask = np.ones_like(yj, dtype=bool)

        yj_valid = yj[mask]
        pj_valid = pj[mask]
        valid_n = int(mask.sum())
        valid_n_list.append(valid_n)

        if valid_n == 0:
            acc_list.append(np.nan)

            macro_f1_list.append(np.nan)
            micro_f1_list.append(np.nan)
            weighted_f1_list.append(np.nan)

            macro_prec_list.append(np.nan)
            micro_prec_list.append(np.nan)
            weighted_prec_list.append(np.nan)

            macro_rec_list.append(np.nan)
            micro_rec_list.append(np.nan)
            weighted_rec_list.append(np.nan)

            n_classes_list.append(0)
            confusion_matrices[target_names[j]] = np.zeros((0, 0), dtype=int)
            continue

        # classification metrics
        acc = accuracy_score(yj_valid, pj_valid)

        macro_f1 = f1_score(yj_valid, pj_valid, average="macro", zero_division=0)
        micro_f1 = f1_score(yj_valid, pj_valid, average="micro", zero_division=0)
        weighted_f1 = f1_score(yj_valid, pj_valid, average="weighted", zero_division=0)

        macro_prec = precision_score(yj_valid, pj_valid, average="macro", zero_division=0)
        micro_prec = precision_score(yj_valid, pj_valid, average="micro", zero_division=0)
        weighted_prec = precision_score(yj_valid, pj_valid, average="weighted", zero_division=0)

        macro_rec = recall_score(yj_valid, pj_valid, average="macro", zero_division=0)
        micro_rec = recall_score(yj_valid, pj_valid, average="micro", zero_division=0)
        weighted_rec = recall_score(yj_valid, pj_valid, average="weighted", zero_division=0)

        n_classes = len(np.unique(yj_valid))
        cm = confusion_matrix(yj_valid, pj_valid)

        acc_list.append(acc)

        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)
        weighted_f1_list.append(weighted_f1)

        macro_prec_list.append(macro_prec)
        micro_prec_list.append(micro_prec)
        weighted_prec_list.append(weighted_prec)

        macro_rec_list.append(macro_rec)
        micro_rec_list.append(micro_rec)
        weighted_rec_list.append(weighted_rec)

        n_classes_list.append(n_classes)
        confusion_matrices[target_names[j]] = cm

    per_target = pd.DataFrame({
        "Accuracy": acc_list,

        "Macro_Precision": macro_prec_list,
        "Macro_Recall": macro_rec_list,
        "Macro_F1": macro_f1_list,

        "Micro_Precision": micro_prec_list,
        "Micro_Recall": micro_rec_list,
        "Micro_F1": micro_f1_list,

        "Weighted_Precision": weighted_prec_list,
        "Weighted_Recall": weighted_rec_list,
        "Weighted_F1": weighted_f1_list,

        "n_classes": n_classes_list,
        "Valid_n": valid_n_list
    }, index=target_names)

    # Overall / pooled metrics
    # Flatten and drop NaNs if any (float labels)
    yt_flat = y_true.ravel()
    yp_flat = y_pred.ravel()
    if np.issubdtype(yt_flat.dtype, np.floating) or np.issubdtype(yp_flat.dtype, np.floating):
        mask_flat = np.isfinite(yt_flat) & np.isfinite(yp_flat)
        yt_flat = yt_flat[mask_flat]
        yp_flat = yp_flat[mask_flat]

    if yt_flat.size == 0:
        micro_acc = np.nan
        micro_macro_f1 = np.nan
        micro_weighted_f1 = np.nan
        micro_macro_prec = np.nan
        micro_weighted_prec = np.nan
        micro_macro_rec = np.nan
        micro_weighted_rec = np.nan
    else:
        micro_acc = accuracy_score(yt_flat, yp_flat)

        micro_macro_f1 = f1_score(yt_flat, yp_flat, average="macro")
        micro_weighted_f1 = f1_score(yt_flat, yp_flat, average="weighted")

        micro_macro_prec = precision_score(yt_flat, yp_flat, average="macro", zero_division=0)
        micro_weighted_prec = precision_score(yt_flat, yp_flat, average="weighted", zero_division=0)

        micro_macro_rec = recall_score(yt_flat, yp_flat, average="macro", zero_division=0)
        micro_weighted_rec = recall_score(yt_flat, yp_flat, average="weighted", zero_division=0)

    overall = {
        # macro: average over targets
        "macro_Accuracy": float(np.nanmean(acc_list)),

        "macro_Macro_Precision": float(np.nanmean(macro_prec_list)),
        "macro_Macro_Recall": float(np.nanmean(macro_rec_list)),
        "macro_Macro_F1": float(np.nanmean(macro_f1_list)),

        "macro_Weighted_Precision": float(np.nanmean(weighted_prec_list)),
        "macro_Weighted_Recall": float(np.nanmean(weighted_rec_list)),
        "macro_Weighted_F1": float(np.nanmean(weighted_f1_list)),

        # micro: pooled across all targets
        "micro_Accuracy": float(micro_acc),

        "micro_Macro_Precision": float(micro_macro_prec),
        "micro_Macro_Recall": float(micro_macro_rec),
        "micro_Macro_F1": float(micro_macro_f1),

        "micro_Weighted_Precision": float(micro_weighted_prec),
        "micro_Weighted_Recall": float(micro_weighted_rec),
        "micro_Weighted_F1": float(micro_weighted_f1),

        "samples": int(N),
        "targets": int(T),
    }

    if round_digits is not None:
        per_target = per_target.round(round_digits)
        overall = {
            k: (round(v, round_digits) if isinstance(v, float) else v)
            for k, v in overall.items()
        }

    return {
        "per_target": per_target,
        "overall": overall,
        "confusion_matrices": confusion_matrices,
        "n_samples": N
    }

def compute_multioutput_metrics_regression(
    y_true: np.ndarray,          # shape (N, T)
    y_pred: np.ndarray,          # shape (N, T)
    target_names: Optional[Sequence[str]] = None,
    tolerances: Optional[Sequence[float]] = None,  # per-target absolute tolerances (same units as target)
    round_digits: Optional[int] = 3
) -> Dict[str, Any]:
    """
    Returns:
      {
        "per_target": pd.DataFrame (rows = targets, cols = metrics),
        "overall": dict of macro/micro metrics and coverage,
        "n_samples": int
      }
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    N, T = y_true.shape
    if target_names is None:
        target_names = [f"y{i}" for i in range(T)]
    if len(target_names) != T:
        raise ValueError("len(target_names) must equal number of targets (T)")

    if tolerances is None:
        tolerances = [np.nan] * T  # no within-tol if not provided
    elif len(tolerances) != T:
        raise ValueError("len(tolerances) must equal number of targets (T)")

    # Mask invalid pairs
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    valid_counts = finite_mask.sum(axis=0)

    # Avoid contaminating metrics with NaNs
    yt = np.where(finite_mask, y_true, np.nan)
    yp = np.where(finite_mask, y_pred, np.nan)

    diff = yp - yt
    abs_err = np.abs(diff)
    sq_err = diff**2

    # Per-target metrics
    mae  = np.nanmean(abs_err, axis=0)
    mse  = np.nanmean(sq_err, axis=0)                 # <-- ADDED
    rmse = np.sqrt(mse)                               # (uses mse for numerical consistency)

    # MAPE safe: exclude 0 or near-zero truths per target
    eps = 1e-12
    mape = np.array([
        np.nanmean((np.abs(yt[:, j]) > eps) * (abs_err[:, j] / (np.abs(yt[:, j]) + eps)) * 100.0)
        for j in range(T)
    ])

    # Bias (mean error): sign shows systematic under/over-prediction
    mbe = np.nanmean(diff, axis=0)

    # R^2 per target (1 - SSE/SST) on available pairs
    r2 = []
    for j in range(T):
        yj = yt[:, j]
        pj = yp[:, j]
        mask = np.isfinite(yj) & np.isfinite(pj)
        if mask.sum() < 2:
            r2.append(np.nan)
            continue
        sse = np.sum((pj[mask] - yj[mask])**2)
        sst = np.sum((yj[mask] - np.mean(yj[mask]))**2)
        r2.append(1.0 - sse / sst if sst > 0 else np.nan)
    r2 = np.array(r2)

    # Pearson correlation per target
    corr = []
    for j in range(T):
        yj = yt[:, j]
        pj = yp[:, j]
        mask = np.isfinite(yj) & np.isfinite(pj)
        if mask.sum() < 2:
            corr.append(np.nan)
            continue
        yjm = yj[mask] - np.mean(yj[mask])
        pjm = pj[mask] - np.mean(pj[mask])
        denom = (np.sqrt(np.sum(yjm**2)) * np.sqrt(np.sum(pjm**2)))
        corr.append(np.sum(yjm * pjm) / denom if denom > 0 else np.nan)
    corr = np.array(corr)

    # Within-tolerance rate per target (fraction of pairs with |error| <= tol)
    within_tol = []
    for j in range(T):
        tol = tolerances[j]
        if not np.isfinite(tol):
            within_tol.append(np.nan)
            continue
        mask = np.isfinite(abs_err[:, j])
        if mask.sum() == 0:
            within_tol.append(np.nan)
            continue
        within_tol.append(np.mean(abs_err[mask, j] <= tol))
    within_tol = np.array(within_tol)

    per_target = pd.DataFrame({
        "MAE": mae,
        "MSE": mse,                                  # <-- ADDED
        "RMSE": rmse,
        "MAPE_%": mape,
        "Bias": mbe,
        "R2": r2,
        "Pearson_r": corr,
        "Within_tol": within_tol,
        "Valid_n": valid_counts
    }, index=target_names)

    # Overall summaries
    overall = {
        # Macro: average of per-target metrics
        "macro_MAE": float(np.nanmean(mae)),
        "macro_MSE": float(np.nanmean(mse)),         # <-- ADDED
        "macro_RMSE": float(np.nanmean(rmse)),
        "macro_MAPE_%": float(np.nanmean(mape)),
        "macro_Bias": float(np.nanmean(mbe)),
        "macro_R2": float(np.nanmean(r2)),
        "macro_Pearson_r": float(np.nanmean(corr)),
        "macro_Within_tol": float(np.nanmean(within_tol)),

        # Micro: compute across all targets pooled
        "micro_MAE": float(np.nanmean(np.abs(diff))),
        "micro_MSE": float(np.nanmean(diff**2)),     # <-- ADDED
        "micro_RMSE": float(np.sqrt(np.nanmean(diff**2))),

        # Coverage
        "overall_valid_pairs": int(np.isfinite(diff).sum()),
        "samples": int(N),
        "targets": int(T),
    }

    if round_digits is not None:
        per_target = per_target.round(round_digits)
        overall = {k: (round(v, round_digits) if isinstance(v, float) else v) for k, v in overall.items()}

    return {"per_target": per_target, "overall": overall, "n_samples": N}

def cf_metrics(
    expected_performance: np.ndarray,
    predicted: np.ndarray,
    original_inputs: pd.DataFrame,
    modified_inputs: pd.DataFrame,
    target_cols: List[str],
    immutable_columns: List[str],
    training_df: pd.DataFrame,
    ML_task_type: str,
    *,
    proximity_norm: str = "l1",
    scaled_validity=False,
    tolerance: float = 0.05,
    reg_validity_eps: float = 0.1,
    feature_weights: Optional[Dict[str, float]] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Compute counterfactual metrics for CLASSIFICATION / REGRESSION:

    - Validity:
        * classification: 1 if predicted == expected_performance
        * regression: 1 if mean abs(target_delta_scaled) <= reg_validity_eps
          where target is min-max scaled using training_df[target_cols]
    - Proximity:
        L1/L2 distance between original and modified inputs (MUTABLE cols only),
        computed in SCALED feature space where each mutable column is
        min-max scaled using statistics from training_df.
    - Sparsity:
        # of mutable features changed (abs(delta_scaled) > tolerance)

    Notes
    -----
    - Min-max is computed per mutable column:
        scaled = (x - min_train) / (max_train - min_train)
      using the column in training_df with the same name.
    - Min-max for regression validity is computed per target column,
      using training_df[target_cols].
    - Only mutable columns are used for proximity/sparsity.
    - This assumes all mutable columns and targets are numeric.
    """

    if training_df is None:
        raise ValueError("training_df must be provided for min-max scaling.")

    ML_task_type = ML_task_type.lower()
    if ML_task_type not in {"classification", "regression"}:
        raise ValueError("ML_task_type must be 'classification' or 'regression'.")

    # --- basic checks on inputs ---
    if original_inputs.shape != modified_inputs.shape:
        raise ValueError("original_inputs and modified_inputs must have the same shape.")
    if not original_inputs.columns.equals(modified_inputs.columns):
        raise ValueError("original_inputs and modified_inputs must have identical columns/order.")

    n = len(original_inputs)

    exp = np.asarray(expected_performance)
    pred = np.asarray(predicted)

    # handle 1D vs 2D for targets
    if exp.ndim == 1:
        exp = exp.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    if exp.shape[0] != n or pred.shape[0] != n:
        raise ValueError(f"expected_performance and predicted must have length N={n}.")
    if exp.shape != pred.shape:
        raise ValueError("expected_performance and predicted must have the same shape.")

    # --- validity ---
    if ML_task_type == "classification":
        # assume single target column for classification (or encoded labels)
        exp_flat = exp.reshape(-1)
        pred_flat = pred.reshape(-1)
        valid = (pred_flat == exp_flat).astype(int)
        validity_rate = float(valid.mean())

    elif scaled_validity==True:  # regression with scaled target-based validity
        if not isinstance(target_cols, (list, tuple)) or len(target_cols) != exp.shape[1]:
            raise ValueError(
                f"target_cols must be a list of length {exp.shape[1]} for regression. "
                f"Got {target_cols}."
            )

        # Make sure training_df has these target columns
        missing_targets = [c for c in target_cols if c not in training_df.columns]
        if missing_targets:
            raise ValueError(
                f"training_df is missing target columns needed for regression validity: "
                f"{missing_targets}"
            )

        train_targets = training_df[target_cols].copy()

        # numeric only
        non_numeric_targets = [
            c for c in target_cols if not pd.api.types.is_numeric_dtype(train_targets[c])
        ]
        if non_numeric_targets:
            raise TypeError(
                "training_df target columns must be numeric for min-max scaling. "
                f"Non-numeric targets: {non_numeric_targets}"
            )

        train_targets = train_targets.astype(float)
        y_min = train_targets.min(axis=0)  # Series indexed by target_cols
        y_max = train_targets.max(axis=0)
        y_denom = (y_max - y_min).replace(0, 1.0)  # avoid division by zero

        # Put exp/pred into DataFrames so broadcasting works by column name
        exp_df = pd.DataFrame(exp, columns=target_cols, index=original_inputs.index)
        pred_df = pd.DataFrame(pred, columns=target_cols, index=original_inputs.index)

        # Min-max scale target values
        exp_scaled = (exp_df - y_min) / y_denom
        pred_scaled = (pred_df - y_min) / y_denom

        # delta in scaled target space
        delta_y_scaled = pred_scaled - exp_scaled   # DataFrame (n, T)

        # mean absolute scaled error per sample → in [0,1]
        err_scaled = np.abs(delta_y_scaled.to_numpy()).mean(axis=1)

        # validity in regression: 1 if scaled error <= reg_validity_eps
        valid = (err_scaled <= float(reg_validity_eps)).astype(int)
        validity_rate = float(valid.mean())
    
    else:
        if not isinstance(target_cols, (list, tuple)) or len(target_cols) != exp.shape[1]:
            raise ValueError(
                f"target_cols must be a list of length {exp.shape[1]} for regression. "
                f"Got {target_cols}."
            )

        # Put exp/pred into DataFrames so broadcasting works by column name
        exp_df = pd.DataFrame(exp, columns=target_cols, index=original_inputs.index).astype(float)
        pred_df = pd.DataFrame(pred, columns=target_cols, index=original_inputs.index).astype(float)

        # relative tolerance per target value
        pct = float(reg_validity_eps)
        if pct < 0:
            raise ValueError("reg_validity_pct must be non-negative (e.g., 0.05 for 5%).")

        tiny = 1e-12  # avoids zero-division when exp == 0
        allowed = pct * np.maximum(np.abs(exp_df), tiny)  # per-element tolerance
        err_abs = (pred_df - exp_df).abs()

        # validity per sample: all targets must be within allowed tolerance
        within = (err_abs <= allowed)
        valid = within.all(axis=1).astype(int).to_numpy()
        validity_rate = float(valid.mean())

    # --- mutable columns only for proximity/sparsity ---
    imm_set = set(immutable_columns)
    missing = [c for c in immutable_columns if c not in original_inputs.columns]
    if missing:
        raise ValueError(f"immutable_columns contains unknown columns: {missing}")

    mutable_cols = [c for c in original_inputs.columns if c not in imm_set]
    if not mutable_cols:
        raise ValueError("No mutable columns remain after removing immutable_columns.")

    x0 = original_inputs[mutable_cols].copy()
    xcf = modified_inputs[mutable_cols].copy()

    # numeric only (for original inputs)
    non_numeric = [c for c in mutable_cols if not pd.api.types.is_numeric_dtype(x0[c])]
    if non_numeric:
        raise TypeError(
            f"All mutable columns must be numeric for proximity/sparsity. Non-numeric: {non_numeric}"
        )

    # --- min-max scaling using training_df for features ---
    missing_train = [c for c in mutable_cols if c not in training_df.columns]
    if missing_train:
        raise ValueError(
            f"training_df is missing mutable columns needed for scaling: {missing_train}"
        )

    train_num = training_df[mutable_cols].copy()

    non_numeric_train = [
        c for c in mutable_cols if not pd.api.types.is_numeric_dtype(train_num[c])
    ]
    if non_numeric_train:
        raise TypeError(
            "training_df mutable columns must be numeric for min-max scaling. "
            f"Non-numeric in training_df: {non_numeric_train}"
        )

    train_num = train_num.astype(float)
    col_min = train_num.min(axis=0)
    col_max = train_num.max(axis=0)
    denom = (col_max - col_min).replace(0, 1.0)  # avoid division by zero for constant cols

    # Scale original and modified (broadcasting by column name)
    x0_scaled = (x0.astype(float) - col_min) / denom
    xcf_scaled = (xcf.astype(float) - col_min) / denom

    x0v = x0_scaled.to_numpy(dtype=float)
    xcfv = xcf_scaled.to_numpy(dtype=float)
    delta = xcfv - x0v  # (N, M) in scaled feature space

    # --- sparsity (# changed mutable features) ---
    changed = np.abs(delta) > float(tolerance)
    sparsity = changed.sum(axis=1).astype(int)

    # --- proximity (distance over mutable features) ---
    # feature weights
    if feature_weights is None:
        w = np.ones(len(mutable_cols), dtype=float)
    else:
        w = np.array([float(feature_weights.get(c, 1.0)) for c in mutable_cols], dtype=float)
        if np.any(w < 0):
            raise ValueError("feature_weights must be non-negative.")

    # normalize weights so they sum to 1 -> weighted average
    w_sum = w.sum()
    if w_sum == 0:
        raise ValueError("feature_weights must not all be zero.")
    w = w / w_sum

    proximity_norm = proximity_norm.lower()

    # delta is (N, M), features already scaled to [0,1] per column
    if proximity_norm == "l1":
        # weighted mean absolute difference -> in [0,1], then invert so 1 is best
        proximity = (np.abs(delta) * w).sum(axis=1)
        proximity = 1.0 - proximity
    elif proximity_norm == "l2":
        # weighted RMS difference -> in [0,1], then invert
        proximity = np.sqrt(((delta ** 2) * w).sum(axis=1))
        proximity = 1.0 - proximity
    else:
        raise ValueError("proximity_norm must be 'l1' or 'l2'.")

    # clip just in case of minor numerical drift
    proximity = np.clip(proximity, 0.0, 1.0)

    per_sample = pd.DataFrame(
        {
            "valid": valid,
            "proximity": proximity,
            "sparsity": sparsity,
        },
        index=original_inputs.index,
    )

    summary = {
        "validity_rate": validity_rate,
        "proximity_mean": float(np.mean(proximity)),
        "proximity_median": float(np.median(proximity)),
        "sparsity_mean": float(np.mean(sparsity)),
        "sparsity_median": float(np.median(sparsity)),
        "n_samples": int(n),
        "n_mutable_features": int(len(mutable_cols)),
        "mutable_columns": mutable_cols,
        "scaled_space": True,
        "scaler_source": "per-column min-max from training_df",
    }

    return summary, per_sample

def mean_sparsity_scores(
    original_inputs: pd.DataFrame,
    modified_inputs: pd.DataFrame,
    training_df: pd.DataFrame,
    immutable_columns: Optional[Iterable[str]] = None,
    tolerance: float = 0.05,) -> float:
    """
    Compute the mean sparsity score across all rows.
    Sparsity = number of changed mutable features (after min-max scaling).
    """
    if immutable_columns is None:
        immutable_columns = []

    if len(original_inputs) != len(modified_inputs):
        raise ValueError(
            "original_inputs and modified_inputs must have the same number of rows."
        )

    # --- mutable columns ---
    imm_set = set(immutable_columns)
    missing = [c for c in immutable_columns if c not in original_inputs.columns]
    if missing:
        raise ValueError(f"immutable_columns contains unknown columns: {missing}")

    mutable_cols = [c for c in original_inputs.columns if c not in imm_set]
    if not mutable_cols:
        raise ValueError("No mutable columns remain after removing immutable_columns.")

    x0 = original_inputs[mutable_cols].copy()
    xcf = modified_inputs[mutable_cols].copy()

    # numeric only
    non_numeric = [c for c in mutable_cols if not pd.api.types.is_numeric_dtype(x0[c])]
    if non_numeric:
        raise TypeError(
            f"All mutable columns must be numeric. Non-numeric: {non_numeric}"
        )

    # --- scaling stats from training_df ---
    missing_train = [c for c in mutable_cols if c not in training_df.columns]
    if missing_train:
        raise ValueError(
            f"training_df is missing mutable columns: {missing_train}"
        )

    train_num = training_df[mutable_cols].astype(float)

    non_numeric_train = [
        c for c in mutable_cols if not pd.api.types.is_numeric_dtype(train_num[c])
    ]
    if non_numeric_train:
        raise TypeError(
            f"Non-numeric mutable columns in training_df: {non_numeric_train}"
        )

    col_min = train_num.min(axis=0)
    col_max = train_num.max(axis=0)
    denom = (col_max - col_min).replace(0, 1.0)

    x0_scaled = (x0.astype(float) - col_min) / denom
    xcf_scaled = (xcf.astype(float) - col_min) / denom

    delta = xcf_scaled.to_numpy() - x0_scaled.to_numpy()

    # --- sparsity ---
    changed = np.abs(delta) > float(tolerance)
    sparsity_per_row = changed.sum(axis=1)

    return float(sparsity_per_row.mean())
    

    



    


#------------Model based feature importances------------

def F_importance_ground_truth_wrapper(
    model_name: str,
    model_path,
    x_row: Sequence[float],
    feature_names: Sequence[str],
    output_variable_number: int,
    target_cols: Optional[Sequence[str]] = None,
    y_index: Optional[int] = None,
    l1_normalize: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-target feature attribution dicts:
      {
        <target_name_or_index>: { <feature_name>: attribution, ... },
        ...
      }

    If y_index is provided, returns only that target.
    """
    model = Load_ModelFile(model_path)
    # pick runner based on model_name
    name = model_name.strip().lower()
    if name in {"linear", "ridge", "lasso"}:
        runner = lambda i: ridge_feature_attributions(
            model, x_row, feature_names=feature_names, y_index=i, l1_normalize=l1_normalize
        )
    elif name in {"dt", "forest", "randomforest", "rf", "decisiontree"}:
        runner = lambda i: tree_feature_attributions(
            model, x_row, feature_names=feature_names, y_index=i, l1_normalize=l1_normalize
        )
    elif name in {"mlp", "transformer", "tf", "keras", "nn"}:
        runner = lambda i: keras_gradxinput_attributions(
            model, x_row, feature_names=feature_names, y_index=i, l1_normalize=l1_normalize
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'")

    # helper to name targets
    def _tname(i: int) -> str:
        if target_cols is not None and 0 <= i < len(target_cols):
            return str(target_cols[i])
        return f"target_{i}"

    results: Dict[str, Dict[str, float]] = {}

    if y_index is not None:
        if not (0 <= y_index < output_variable_number):
            raise IndexError(f"y_index {y_index} out of range [0, {output_variable_number-1}]")
        results[_tname(y_index)] = runner(y_index)
        return results

    # all targets 0..M-1
    for i in range(output_variable_number):
        results[_tname(i)] = runner(i)

    return results
    
    

def ridge_feature_attributions(
    model,                          # e.g., sklearn.linear_model.Ridge
    x_row: Sequence[float],
    feature_names: Sequence[str],
    y_index: int = 0,
    l1_normalize: bool = True
) -> Dict[str, float]:
    """
    Returns {feature: attribution} for Ridge/Linear models.
    Multi-output safe.
    """
    x = np.atleast_2d(np.asarray(x_row, dtype=np.float32))
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise ValueError("Model has no coef_. Did you pass a linear/Ridge model?")

    coef = np.asarray(coef, dtype=np.float32)
    if coef.ndim == 1:
        beta = coef
    elif coef.ndim == 2:
        beta = coef[y_index]
    else:
        raise ValueError(f"Unexpected coef_ shape {coef.shape}")

    attrib = (x.flatten() * beta.astype(np.float32))
    if l1_normalize:
        denom = np.sum(np.abs(attrib)) + 1e-12
        attrib = attrib / denom

    return {fname: round(float(val), 2) for fname, val in zip(feature_names, attrib.tolist())}


def tree_feature_attributions(
    model,                          # DecisionTreeRegressor or RandomForestRegressor
    x_row: Sequence[float],
    feature_names: Sequence[str],
    y_index: int = 0,
    l1_normalize: bool = True
) -> Dict[str, float]:
    """
    Returns {feature: attribution} for tree/forest models.
    Multi-output safe via y_index when supported by sklearn.
    """
    x = np.atleast_2d(np.asarray(x_row, dtype=np.float32))

    def _single_tree_attrib(tree, x) -> np.ndarray:
        T = tree.tree_
        feature = T.feature
        value = T.value

        node_indicator = tree.decision_path(x).indices
        if value.ndim == 3:
            vals = value[:, 0, :]
            node_vals = vals[:, y_index]
        elif value.ndim == 2:
            node_vals = value[:, y_index]
        else:
            node_vals = value[:, 0]

        contrib = np.zeros(x.shape[1], dtype=np.float32)
        last_val = node_vals[node_indicator[0]]
        for n in node_indicator[1:]:
            cur_val = node_vals[n]
            delta = cur_val - last_val
            last_val = cur_val
            fidx = feature[n]
            if fidx >= 0:
                contrib[fidx] += delta
        return contrib

    if hasattr(model, "tree_"):
        attrib = _single_tree_attrib(model, x)
    elif hasattr(model, "estimators_"):
        acc = np.zeros(x.shape[1], dtype=np.float32)
        for est in model.estimators_:
            acc += _single_tree_attrib(est, x)
        attrib = acc / max(len(model.estimators_), 1)
    else:
        raise ValueError("Unsupported tree model type.")

    if l1_normalize:
        denom = np.sum(np.abs(attrib)) + 1e-12
        attrib = attrib / denom

    return {fname: round(float(val), 2) for fname, val in zip(feature_names, attrib.tolist())}


def keras_gradxinput_attributions(
    model,                          # tf.keras.Model
    x_row: Sequence[float],
    feature_names: Sequence[str],
    y_index: int = 0,
    l1_normalize: bool = True
) -> Dict[str, float]:
    import tensorflow as tf

    x_np = np.atleast_2d(np.asarray(x_row, dtype=np.float32))
    x_tf = tf.Variable(tf.convert_to_tensor(x_np))

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        y_pred = model(x_tf, training=False)
        if len(y_pred.shape) == 1 or y_pred.shape[-1] == 1:
            target = y_pred if len(y_pred.shape) == 1 else y_pred[:, 0]
        else:
            target = y_pred[:, y_index]

    grads = tape.gradient(target, x_tf).numpy().reshape(-1)
    attrib = grads * x_np.reshape(-1)

    if l1_normalize:
        denom = np.sum(np.abs(attrib)) + 1e-12
        attrib = attrib / denom

    return {fname: round(float(val), 2) for fname, val in zip(feature_names, attrib.tolist())}

#-------------Feature importance helpers-------------
def _to_2d_row(x_row: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("x_row must be 1D; got shape {}".format(arr.shape))
    return arr

def _l1_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sum(np.abs(vec)) + eps
    return vec / s
    