# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:19:11 2025

@author: arnab
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import List, Iterable, Optional, Union, Sequence, Dict, Any, Tuple
from scipy.stats import norm
import re
from Expert_classes import SVR_expert, OLS_expert
from Datapoint_search_algorithms import KNN_similar_points
import skfuzzy as fuzz
from scipy.stats import chi2
import numpy.linalg as la

from Cat_variable_functions import fit_cat_encoder, transform_cat_encoder, encode_query_classification, Encode_all


# ---------------------------------- Main Expert creator class--------------------------------------
class Clustering_classifier:
    def __init__(self, metric, input_cols, ground_truth_cols, prediction_cols, immutables):

        self.metric = metric
        self.input_cols = input_cols
        self.ground_truth_cols = ground_truth_cols
        self.prediction_cols = prediction_cols
        self.Immutables= immutables

        self.Clusterer = None
        self.expert = None
        self.Clusterer_obj = None
        self.Clustered_Df = None
        self.Cluster_keys = None
        self.training_enabled = False
        self.Clustering_cols = None

    def Create_Clusters(
        self,
        Train_dataframe,
        clustering_cols,
        metric,
        X_cat_encoder=None,
        Y_cat_encoder=None,
        Clusterer="Gaussian",
        expert="SVR",
        path="",
        max_clusters=5
    ):
        """Two tasks: Create clusters and cluster keys from Data frame and generate description for each cluster"""
        self.Clustering_cols = clustering_cols
        self.Clusterer = Clusterer
        self.expert = expert
        self.X_cat_encoder=X_cat_encoder
        self.Y_cat_encoder=Y_cat_encoder

        self.Clusterer_obj = self._clusterer_object_creator(self.Clusterer)
        Clustered_dataframe = self.Clusterer_obj.Create_clusters(
            Train_dataframe, self.Clustering_cols, metric, X_cat_encoder=X_cat_encoder,Y_cat_encoder=Y_cat_encoder, K_max=max_clusters, Prob_cols=True
        )

        self.Clustered_Df = Clustered_dataframe

        self.Cluster_text_descriptions = self.Clusterer_obj.Describe_clusters(self.Clustered_Df, X_cat_encoder=X_cat_encoder)

        self.Cluster_characteristics = self._expert_builder(Clustered_dataframe, X_cat_encoder=X_cat_encoder)

    def Retreive_Cluster_expert_answers(self, query_X, query_Y, output_upgrade=None):
        """Retreive string descriptions of clusters similar to query"""
        
        if (self.Y_cat_encoder!=None):
            if self.metric=='Accuracy':
                q_df=encode_query_classification(query_X)
                q_encoded = transform_cat_encoder(q_df, self.X_cat_encoder)
                query=self._df_row_to_query_string(q_encoded.iloc[0])
                Query_list = self._extract_list_of_numbers(query)
            elif self.metric=='Counterfactuals':
                
                q_df=encode_query_classification(query_X+', '+query_Y)
                df_query = q_df[self.Immutables + self.ground_truth_cols]
                df_query_encoded=Encode_all(df_query, self.X_cat_encoder, self.Y_cat_encoder)
                Query_list = self._extract_list_of_numbers(df_query_encoded)

        else:
            if self.metric=='Accuracy':
                Query_list = self._extract_list_of_numbers(query_X)
            elif self.metric=='Counterfactuals':
                Upgraded_features_list = self._extract_and_upgrade_list(query_Y, output_upgrade)
                Immutable_dict=self.extract_immutable_numbers(query_X)
                Immutable_list= list(Immutable_dict.values())
                Query_list = Immutable_list + Upgraded_features_list
            
        

        assigned_clusters, Cluster_membership_string = self._find_Similar_Clusters(Query_list)

        All_cluster_answers = Cluster_membership_string
        
        #extracting every cluster description
        for Cluster_id in assigned_clusters:
            Cluster_text_description = self.Cluster_text_descriptions[Cluster_id]

            #Cluster_gradient_description = self.Cluster_characteristics[Cluster_id]["Gradients"]
            Cluster_gradient_description =''
            if(self.metric=='Accuracy'):
                Similar_points_string = self._find_Similar_Points(Query_list, Cluster_id, X_cat_encoder=self.X_cat_encoder, Y_cat_encoder=self.Y_cat_encoder, Delta_string=False)
            else:
                Similar_points_string = self._find_Similar_Points(Query_list, Cluster_id, X_cat_encoder=self.X_cat_encoder, Y_cat_encoder=self.Y_cat_encoder, Delta_string=False)

            
            Cluster_Expert_answer = (
                Cluster_text_description
                + "\n\n"
                + Cluster_gradient_description
                + "\n\n"
                + Similar_points_string
                + "\n\n"
            )
            
            

            All_cluster_answers = All_cluster_answers + Cluster_Expert_answer

        return All_cluster_answers

    def _clusterer_object_creator(self, Clusterer_name: str):
        if Clusterer_name == "Gaussian":
            Clusterer_obj = Gaussian_clusterer()
        elif Clusterer_name == "Fuzzy":
            Clusterer_obj = FuzzyCMeans_clusterer()
        else:
            print(
                f"Mistake in name of cluster classifier, please recheck, there is no {Clusterer_name}"
            )

        return Clusterer_obj
    
    def _df_row_to_query_string(self, df_row: pd.Series) -> str:
        parts = []
        for col, val in df_row.items():
            parts.append(f"{col}: {val}")
        return ", ".join(parts)
    

    def _extract_list_of_numbers(self, input_obj):
        """
        If input is a string:
            - Extract numbers using regex (like before).

        If input is a DataFrame:
            - Flatten all values to a list of floats (row-major order).

        Returns:
            list[float]
        """
        # Case 1: input is a single STRING (keep your original logic)
        if isinstance(input_obj, str):
            numbers = re.findall(r":\s*(-?\d+(?:\.\d+)?)", input_obj)
            return [float(num) for num in numbers]

        # Case 2: input is a DataFrame -> flatten values
        elif isinstance(input_obj, pd.DataFrame):
            # If you only ever expect one row, this is perfect:
            return input_obj.astype(float).values.ravel().tolist()

            # If you want to be explicit about first row only, use:
            # return input_obj.iloc[0].astype(float).values.tolist()

        # Anything else is an error
        else:
            raise TypeError("Input must be either a string or a pandas DataFrame.")


    def _extract_and_upgrade_list(self, input_str, output_upgrade):
        """
        Extract numbers from a string and add corresponding upgrade values.

        Parameters
        ----------
        input_str : str
            String containing numbers (e.g. "RSRP -95, DL 2300, SNR 10").
        output_upgrade : list[float]
            List of values to add to extracted numbers (must match length).

        Returns
        -------
        list[float]
            Extracted numbers after applying upgrades.
        """
        # Extract all numbers (ints, floats, negatives)
        numbers = re.findall(r"-?\d+\.?\d*", input_str)
        numbers = [float(num) for num in numbers]

        # If lengths don’t match, raise error
        if len(numbers) != len(output_upgrade):
            raise ValueError(
                f"Length mismatch: extracted {len(numbers)} numbers, "
                f"but got {len(output_upgrade)} upgrades."
            )

        # Apply upgrades elementwise
        upgraded = [n + u for n, u in zip(numbers, output_upgrade)]
        return upgraded
    
    def extract_immutable_numbers(self, text: str) -> Dict[str, float]:
        """
        Extract numeric values from `text` for each feature name in self.immutables.

        Supports patterns like:
            age=42
            age: 42
            age = 42.5
        Numbers can be ints, floats, +/- signs, and scientific notation.

        Returns a dict: {feature_name: number}
        Only features that are found in the string will appear in the result.
        """
        result: Dict[str, float] = {}

        for col in self.Immutables:
            # Regex: column name, optional spaces, ':' or '=', spaces, then a number
            pattern = rf"{re.escape(col)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

            match = re.search(pattern, text)
            if match:
                value_str = match.group(1)
                try:
                    result[col] = float(value_str)
                except ValueError:
                    # If it somehow isn't a valid float, just skip it
                    continue

        return result

    def _find_Similar_Clusters(self, query, prob_threshold: float = 0.1):
        """Search clusters based on key value and query, sorted by descending probability."""
        # memberships/probs for this query
        clusters = self.Clusterer_obj.Find_experts(query, self.Cluster_keys)
        probs = clusters["probs"]
    
        # cluster prevalence from the whole clustered_df (global share)
        counts = self.Clustered_Df["Cluster_id"].value_counts(normalize=True)
    
        # sort (cid, p) pairs by probability descending
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    
        # collect clusters above threshold
        assigned_clusters = [cid for cid, p in sorted_probs if p >= prob_threshold]
    
        # build explanation in descending order
        explanation_parts = []
        for cid, p in sorted_probs:
            if p >= prob_threshold:
                share = counts.get(cid, 0.0) * 100
                explanation_parts.append(
                    f"cluster {cid} ({share:.1f}% of all global points) with probability {p*100:.1f}%"
                )
    
        if explanation_parts:
            explanation = (
                "\nThe point belongs to "
                + " and ".join(explanation_parts)
                + ".\n\n"
            )
        else:
            explanation = "\nNo clusters found above the probability threshold.\n"
    
        return assigned_clusters, explanation
    
    


    def _find_Similar_Points(self, query, Cluster_id, num_of_points=15, X_cat_encoder=None,Y_cat_encoder=None, Delta_string=False):
        """Search clusters for similar points to query using functions which index and search similar points"""
        
        neighbors = KNN_similar_points(
            query,
            self.Clustered_Df,
            feature_cols=self.Clustering_cols,
            nearby_clusters=Cluster_id,
            X_cat_encoder=X_cat_encoder,
            Y_cat_encoder=Y_cat_encoder,
            k=num_of_points,
            use_overlapping=True,  # also include rows whose Experts contains 2 or 5
            return_cols=[
                "Y_true_DL",
                "Y_pred_DL",
                "timestamp",
            ],  # optional extra context
        )
        
        neighbors.rename(columns={c: f"{c}_ground_truth" for c in set(neighbors.columns) & set(self.ground_truth_cols)},inplace=True)
        ground_truth_col_labels = [col + "_ground_truth" for col in self.ground_truth_cols]
        if Delta_string==True:
            Similar_points_string=self.make_delta_evidence_baseline_first(neighbors, input_cols=self.input_cols, 
                                                                          ground_truth_cols=ground_truth_col_labels, 
                                                                          prediction_cols=self.prediction_cols)
        
        else:
            Similar_points_string = (
                f"Cluster {Cluster_id} similar points to query: \n\n"
                + neighbors.to_string(index=False).replace("\n", "\n\n"))
 
            

        return Similar_points_string

    def _expert_builder(self, Clustered_dataframe, X_cat_encoder=None):
        if self.expert == "SVR":
            expert = SVR_expert(self.input_cols, self.prediction_cols)
        elif self.expert == "OLS":
            expert = OLS_expert(self.input_cols, self.prediction_cols)
        else:
            print(f"Expert not found or typo error: {self.expert}")

        expert.Build_expert(Clustered_dataframe, cluster_col="Cluster_id")
        metrics = expert.Cluster_metrics(Clustered_dataframe, X_cat_encoder=X_cat_encoder)

        return metrics
    
    def make_delta_evidence_baseline_first(
        self,
        neighbors: pd.DataFrame,
        input_cols: list,             # feature columns (order shown in Δinputs)
        ground_truth_cols: list,      # e.g., ["RSRP_ground_truth","DL_bitrate_ground_truth"]
        prediction_cols: list,        # e.g., ["RSRP_predicted","DL_bitrate_predicted"]
        max_rows: int | None = None,
        eps: float = 1e-6,
        show_baseline: bool = True,
        max_unique_for_categorical: int = 10) -> str:
        """
        Build lines like:
          '• ΔRSRQ:+1.00, ΔSNR:+2.00 → ΔRSRP_ground_truth:+5.10, ΔDL_bitrate_ground_truth:+2100.00
             → ΔRSRP_predicted:+5.00, ΔDL_bitrate_predicted:+2050.00 (model_error=RSRP:0.10, DL_bitrate:50.00)'
        using the FIRST ROW of `neighbors` as the baseline.

        - Deltas are only computed for **numeric, non-categorical** columns.
        - Categorical columns are auto-detected from `neighbors` and **ignored** in Δ and model_error.
          Heuristic for categorical:
            * non-numeric dtypes (object/string/category/bool), OR
            * numeric columns with <= max_unique_for_categorical distinct values.
        """

        if neighbors is None or len(neighbors) == 0:
            return (
                "Evidence (Δinputs→Δground_truth→Δmodel_outputs (model_error=|ground_truth−model_outputs|)):\n"
                "• (no neighbors provided)"
            )

        # Use first row as baseline
        base = neighbors.iloc[0]

        # -------- auto-detect categorical columns from neighbors --------
        # Only consider columns that actually appear in our three groups
        candidate_cols = list(
            set(input_cols) | set(ground_truth_cols) | set(prediction_cols)
        )
        candidate_cols = [c for c in candidate_cols if c in neighbors.columns]
        if candidate_cols:
            subdf = neighbors[candidate_cols]
        else:
            subdf = neighbors.iloc[:, 0:0]  # empty frame, just in case

        # non-numeric are categorical by default
        non_numeric = subdf.select_dtypes(exclude=["number"]).columns

        # numeric columns with low cardinality are treated as categorical as well
        numeric = subdf.select_dtypes(include=["number"]).columns
        low_card_numeric = [
            c for c in numeric
            if subdf[c].nunique(dropna=True) <= max_unique_for_categorical
        ]

        categorical_cols = set(non_numeric) | set(low_card_numeric)
        # everything else is considered numeric-for-deltas

        # ----------------------------------------------------------------

        # Helper: baseline key=value line (we keep all columns here, even categorical)
        def _kv_line(cols):
            return ", ".join(f"{c}:{base[c]}" for c in cols if c in neighbors.columns)

        baseline_parts = []
        if show_baseline:
            bi = _kv_line(input_cols)
            bg = _kv_line(ground_truth_cols)
            bp = _kv_line(prediction_cols)
            baseline_line = "Baseline row: "
            sub = []
            if bi:
                sub.append(f"inputs=({bi})")
            if bg:
                sub.append(f"ground_truth=({bg})")
            if bp:
                sub.append(f"outputs=({bp})")
            baseline_parts = [baseline_line + " | ".join(sub), ""]

        header = (
            "-----Similar points to query from Cluster---------:\n"
            + (" ".join(baseline_parts) if baseline_parts else "")
            + "\nAll deltas (Δ) are computed relative to this baseline row.\n"
            + "We show a chained map: Δinputs → Δground_truth → Δmodel_outputs (predictions).\n"
            "Example: ΔUL_bitrate:-30 → ΔDL_bitrate_ground_truth:+2100 → ΔDL_bitrate_predicted:+2050 (model_error=DL_bitrate:150)\n\n"
            "Evidence (Δinputs→Δground_truth→Δmodel_outputs (model_error=|ground_truth−model_outputs|)):"
        )

        # Map ground_truth col → corresponding prediction col via naming convention
        gt_to_pred = {}
        for gt_col in ground_truth_cols:
            base_name = gt_col.replace("_ground_truth", "")
            pred_col = base_name + "_predicted"
            if pred_col in prediction_cols:
                gt_to_pred[gt_col] = pred_col

        # Iterate remaining rows as evidence (skip baseline itself)
        it = neighbors.iloc[1:].itertuples(index=False)
        if max_rows is not None:
            it = (row for _, row in zip(range(max_rows), it))

        nb_cols = list(neighbors.columns)
        lines = []

        for row in it:
            rowd = row._asdict() if hasattr(row, "_asdict") else dict(zip(nb_cols, row))

            # ---------- Δinputs ----------
            din_parts = []
            for c in input_cols:
                if c in categorical_cols:
                    continue  # ignore cats completely
                if c not in rowd or c not in base:
                    continue

                try:
                    d = float(rowd[c]) - float(base[c])
                except (TypeError, ValueError):
                    continue

                if abs(d) > eps:
                    din_parts.append(f"Δ{c}:{d:+.2f}")

            # If no numeric input change vs baseline -> skip this neighbor entirely
            if not din_parts:
                continue

            # ---------- Δground_truth ----------
            dgt_parts, dgt_map = [], {}
            for c in ground_truth_cols:
                if c in categorical_cols:
                    continue
                if c not in rowd or c not in base:
                    continue

                try:
                    d = float(rowd[c]) - float(base[c])
                except (TypeError, ValueError):
                    continue

                if abs(d) > eps:
                    dgt_parts.append(f"Δ{c}:{d:+.2f}")
                    dgt_map[c] = d

            # ---------- Δpredictions ----------
            dout_parts, dout_map = [], {}
            for c in prediction_cols:
                if c in categorical_cols:
                    continue
                if c not in rowd or c not in base:
                    continue

                try:
                    d = float(rowd[c]) - float(base[c])
                except (TypeError, ValueError):
                    continue

                if abs(d) > eps:
                    dout_parts.append(f"Δ{c}:{d:+.2f}")
                    dout_map[c] = d

            # ---------- model_error (numeric targets only) ----------
            err_pairs = []
            for gt_col, pred_col in gt_to_pred.items():
                if gt_col in categorical_cols or pred_col in categorical_cols:
                    continue  # skip categorical targets in numeric error
                if gt_col in dgt_map and pred_col in dout_map:
                    err_val = abs(dgt_map[gt_col] - dout_map[pred_col])
                    if err_val > eps:
                        short_name = gt_col.replace("_ground_truth", "")
                        err_pairs.append(f"{short_name}:{err_val:.2f}")

            # Assemble line
            rhs = []
            if dgt_parts:
                rhs.append(", ".join(dgt_parts))
            if dout_parts:
                rhs.append(", ".join(dout_parts))

            line = "• " + ", ".join(din_parts)
            if rhs:
                line += " → " + " → ".join(rhs)
            if err_pairs:
                line += " (model_error=" + ", ".join(err_pairs) + ")"

            lines.append(line)

        body = "\n".join(lines) if lines else "• (no non-zero deltas vs baseline)"
        return header + "\n" + body




    

# ---------------------------------- Clustering Algorithms--------------------------------------
class Clusterer_template(ABC):
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None

    @abstractmethod
    def Create_clusters(self, Train_dataframe, feature_cols, path=""):
        """Create clusters and return a dataframe with a new column 'Cluster id"""
        pass

    @abstractmethod
    def Find_experts(self, query):
        """Find clusters/experts nearest to a given query"""
        pass

    @abstractmethod
    def Describe_clusters(self):
        """Rich english description of each cluster"""
        pass


class Gaussian_clusterer(Clusterer_template):
    def __init__(self):
        self.gmm = None  # fitted gmm model
        self.scaler = None  # fitted scaler
        self.feature_cols = None

    def Create_clusters(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        metric,
        K_max=11,
        *,
        X_cat_encoder=None,
        Y_cat_encoder=None,
        criterion: str = "bic",  # "bic" or "aic" for model selection
        scale: bool = True,  # standardize features before GMM?
        covariance_type: str = "full",  # "full" | "tied" | "diag" | "spherical"
        reg_covar: float = 1e-5,  # numeric stability for covariances
        n_init: int = 5,  # GMM restarts (helps avoid bad local minima)
        random_state: int = 42,
        # Overlapping-expert assignment controls (soft → selected experts)
        prob_threshold: Optional[float] = None,  # include any cluster with p >= this
        cumprob_threshold: float = 0.8,  # else: take top clusters until cum ≥ this
        max_experts: int = 3,  # cap number of experts per row
        # Output column naming
        Prob_cols=False,
        hard_label_col: str = "Cluster_id",
        soft_prefix: str = "P_cluster_",
    ) -> Tuple[pd.DataFrame, GaussianMixture, Optional[StandardScaler]]:
        """
        Fit a Gaussian Mixture Model over selected columns, auto-select K with BIC/AIC,
        and add both hard and soft cluster assignments to the dataframe. Also produce
        an 'Experts' list per row with up to `max_experts` overlapping experts.

        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe (unchanged; a copy is returned).
        feature_cols : list[str]
            Column names used as features for clustering (e.g., inputs + error features).
        K_range : range
            Candidate numbers of mixture components to try (e.g., range(1, 9)).
        criterion : {"bic","aic"}
            Information criterion to pick the best K.
        scale : bool
            If True, standardize features (recommended).
        covariance_type : {"full","tied","diag","spherical"}
            GMM covariance structure; "diag" often stabilizes high-d settings.
        reg_covar : float
            Small positive value added to covariances for numerical stability.
        n_init : int
            Number of random initializations of GMM.
        random_state : int
            Seed for reproducibility.

        prob_threshold : float or None
            If set, select all clusters with responsibility ≥ threshold. If none pass,
            fall back to cumulative strategy.
        cumprob_threshold : float
            When using cumulative strategy, select top clusters until cumulative
            responsibility ≥ this value.
        max_experts : int
            Maximum experts to keep per row (keeps explanations compact).

        hard_label_col : str
            Name of the hard-label column (argmax cluster id).
        soft_prefix : str
            Prefix for soft-probability columns (e.g., "P_cluster_0", ...).

        Returns
        -------
        output_df : pd.DataFrame
            Copy of input df with:
              - hard label column (Cluster_id by default)
              - soft probability columns (P_cluster_*)
              - 'Experts' list column (overlapping experts per row)
        """
        # -----------------------------
        # (A) Extract and (optionally) standardize features
        # -----------------------------
        K_range = range(1, K_max)
        self.feature_cols = feature_cols
        if (Y_cat_encoder!=None and metric=='Accuracy'):
            df_unencoded=df.copy()
            df_X_encoded=transform_cat_encoder(df_unencoded, X_cat_encoder)
            
            X = df_X_encoded[feature_cols].to_numpy()
        
        elif(Y_cat_encoder!=None and metric=='Counterfactuals'):
            df_unencoded=df.copy()
            df_X_encoded=transform_cat_encoder(df_unencoded, X_cat_encoder)
            df_XY_encoded=transform_cat_encoder(df_X_encoded, Y_cat_encoder)
            
            X = df_XY_encoded[feature_cols].to_numpy()
            
        else:
            X = df[feature_cols].to_numpy()
        
        scaler = None
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scaler = scaler

        # -----------------------------
        # (B) Model selection: sweep K and pick best by BIC/AIC
        # -----------------------------
        best_score = np.inf
        best_gmm = None

        for k in K_range:
            gmm_k = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                n_init=n_init,
                random_state=random_state,
            )
            gmm_k.fit(X)
            score = gmm_k.bic(X) if criterion.lower() == "bic" else gmm_k.aic(X)

            if score < best_score:
                best_score = score
                best_gmm = gmm_k

        gmm = best_gmm  # chosen mixture
        self.gmm = gmm

        # -----------------------------
        # (C) Soft responsibilities and hard labels
        # -----------------------------
        probs = self.gmm.predict_proba(X)  # N x K responsibilities; row sums to 1
        hard_labels = probs.argmax(axis=1)  # argmax cluster id per row

        # -----------------------------
        # (D) Build output dataframe with hard + soft assignments
        # -----------------------------
        output_df = df.copy()
        output_df[hard_label_col] = hard_labels

        # Add columns: P_cluster_0, P_cluster_1, ..., P_cluster_{K-1}
        if Prob_cols == True:
            for j in range(probs.shape[1]):
                output_df[f"{soft_prefix}{j}"] = probs[:, j]

        experts_col = []

        for i in range(len(output_df)):
            row_probs = probs[i]  # shape: (K,)
            ranked = np.argsort(row_probs)[::-1]  # indices sorted by p desc
            selected = []

            if prob_threshold is not None:
                # Fixed threshold first
                selected = [
                    int(j) for j, p in enumerate(row_probs) if p >= prob_threshold
                ]

                # Fallback to cumulative if nothing met the threshold
                if not selected:
                    cum = 0.0
                    tmp = []
                    for j in ranked:
                        tmp.append(int(j))
                        cum += row_probs[j]
                        if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                            break
                    selected = tmp
            else:
                # Pure cumulative strategy
                cum = 0.0
                tmp = []
                for j in ranked:
                    tmp.append(int(j))
                    cum += row_probs[j]
                    if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                        break
                selected = tmp

            # Enforce maximum number of experts (keeps prompts concise)
            selected = selected[:max_experts]

            # Optional: mark as unassigned if using a strict threshold and none pass
            if prob_threshold is not None and len(selected) == 0:
                selected = [-1]

            experts_col.append(selected)

        output_df["Experts"] = experts_col

        return output_df

    def Find_experts(
                self,
                sample,  # dict | pd.Series | 1D np.array
                feature_cols: List[str],
                prob_threshold: Optional[float] = None,  # e.g. 0.25
                cumprob_threshold: float = 0.8,
                max_experts: int = 3,) -> Dict[str, Any]:
                """
                Assign a new sample to GMM experts and report its location INSIDE the assigned
                Gaussian as a chi-square quantile (0..1).
            
                Returns
                -------
                {
                  'Cluster_id': int,               # argmax cluster
                  'probs': np.ndarray,             # shape (K,), responsibilities
                  'Experts': List[int],            # up to max_experts selected via thresholds
                  'Quantile_in_cluster': float,    # chi^2 CDF of Mahalanobis^2 w.r.t. assigned cluster
                  'Mahalanobis2_in_cluster': float # squared Mahalanobis distance to assigned cluster
                }
                """
                # ---- (A) Build 1xD feature vector in the same column order ----
                if isinstance(sample, (pd.Series, dict)):
                    x = np.array([sample[c] for c in feature_cols], dtype=float).reshape(1, -1)
                else:
                    # assume it's already a 1D array matching feature_cols order
                    x = np.asarray(sample, dtype=float).reshape(1, -1)
            
                # ---- (B) Apply scaler if used during training ----
                x_s = self.scaler.transform(x) if getattr(self, "scaler", None) is not None else x
            
                # ---- (C) Soft responsibilities and hard label ----
                probs = self.gmm.predict_proba(x_s)[0]  # shape: (K,)
                hard_label = int(np.argmax(probs))
            
                # ---- (D) Overlapping expert selection (same logic as training) ----
                ranked = np.argsort(probs)[::-1]
                selected: List[int] = []
            
                if prob_threshold is not None:
                    selected = [int(j) for j, p in enumerate(probs) if p >= prob_threshold]
                    if not selected:
                        # fallback to cumulative
                        cum, tmp = 0.0, []
                        for j in ranked:
                            tmp.append(int(j))
                            cum += probs[j]
                            if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                                break
                        selected = tmp
                else:
                    # pure cumulative
                    cum, tmp = 0.0, []
                    for j in ranked:
                        tmp.append(int(j))
                        cum += probs[j]
                        if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                            break
                    selected = tmp
            
                selected = selected[:max_experts]
            
                # ---- (E) Mahalanobis^2 and chi^2-quantile inside the assigned Gaussian ----
                k = hard_label
                diff = (x_s - self.gmm.means_[k])  # shape: (1, D)
                D = diff.shape[1]
            
                # Use precisions_cholesky_ which exists for all covariance types
                P = self.gmm.precisions_cholesky_
            
                # Compute squared Mahalanobis distance m2 = || diff @ P_k^T ||^2
                if P.ndim == 3:
                    # 'full' covariance: (K, D, D)
                    y = diff @ P[k].T
                    m2 = float(np.sum(y * y))
                elif P.ndim == 2:
                    if P.shape[0] == self.gmm.means_.shape[0]:
                        # 'diag' covariance: (K, D) — P holds sqrt(precision) per dim
                        y = diff * P[k]  # elementwise
                        m2 = float(np.sum(y * y))
                    else:
                        # 'tied' covariance: (D, D) shared
                        y = diff @ P.T
                        m2 = float(np.sum(y * y))
                else:
                    # 'spherical' covariance: (K,) — scalar sqrt(precision) per component
                    scale = float(P[k])
                    y = diff * scale
                    m2 = float(np.sum(y * y))
            
                quantile = float(chi2.cdf(m2, df=D))
            
                return {
                    "Cluster_id": hard_label,
                    "probs": probs,
                    "Experts": selected,
                    "Quantile_in_cluster": quantile,
                    "Mahalanobis2_in_cluster": m2,
                }

    def Describe_clusters(self, clustered_df, X_cat_encoder=None):
        cluster_info_df = self._cluster_info(clustered_df, cat_encoders=X_cat_encoder)
        Cluster_description_dict = self._rich_text_cluster_descriptions(cluster_info_df)
        return Cluster_description_dict

    def _cluster_info(
        self,
        global_df: Optional[pd.DataFrame] = None,  
        cat_encoders=None,        # pass full dataset to get global percentiles
        quantiles: Tuple[float, float] = (0.1, 0.9),
        round_to: int = 3,) -> pd.DataFrame:
        """
        Produce per-cluster, per-feature marginal percentile ranges directly from a fitted GMM.
        Optionally appends where the cluster mean sits in the *global* feature distribution
        (as a percentile) if `global_df` is provided.

        Now also supports categorical features via `self.classification_label_encoder`,
        where those features are assumed to have been label-encoded before fitting the GMM.

        Returns
        -------
        DataFrame with columns, e.g.:

          For numeric features:
            ['cluster','feature','type','q_low','q_high','mean','std','global_percentile']

          For categorical features:
            ['cluster','feature','type','mean_code','most_likely_category','category_rank']

        All numeric numbers are in ORIGINAL units if self.scaler is a StandardScaler used during fitting.
        """
        if self.gmm is None:
            raise RuntimeError("GMM not fitted. Fit before calling _cluster_info().")

        K, D = self.gmm.means_.shape
        if D != len(self.feature_cols):
            raise ValueError("feature_cols length must match GMM feature dimension")

        # z-scores for requested quantiles
        ql, qh = quantiles
        zL, zH = norm.ppf([ql, qh])

        # helper: per-component marginal variances (diagonal of covariance)
        def component_variances(k: int) -> np.ndarray:
            ct = self.gmm.covariance_type
            cov = self.gmm.covariances_
            if ct == "full":
                return np.diag(cov[k])
            elif ct == "tied":
                return np.diag(cov)
            elif ct == "diag":
                return cov[k]
            elif ct == "spherical":
                # spherical variance times I → same variance for every feature
                return np.full(D, cov[k])
            else:
                raise ValueError(f"Unsupported covariance_type: {ct}")

        # categorical encoders (optional)
        
        categorical_cols = set(cat_encoders.keys()) if cat_encoders is not None else set()

        rows = []
        for k in range(K):
            mu_k = self.gmm.means_[k].copy()         # model space (possibly scaled)
            var_k = component_variances(k).copy()
            std_k = np.sqrt(np.maximum(var_k, 1e-12))

            # map back to original scale if StandardScaler was used
            mu_orig = mu_k.copy()
            std_orig = std_k.copy()
            if getattr(self, "scaler", None) is not None:
                # x_orig = x_scaled * scale_ + mean_
                mu_orig = mu_k * self.scaler.scale_ + self.scaler.mean_
                std_orig = std_k * self.scaler.scale_

            for j, feat in enumerate(self.feature_cols):

                # --------- CATEGORICAL FEATURES ----------
                if feat in categorical_cols and cat_encoders is not None:
                    le = cat_encoders[feat]

                    # Use mu_orig[j] as "mean code" in original (label-encoded) space
                    mean_code = float(mu_orig[j])

                    classes = le.classes_
                    class_indices = np.arange(len(classes))

                    # distance of cluster mean to each category index
                    distances = np.abs(class_indices - mean_code)
                    order = np.argsort(distances)

                    most_likely_cat = classes[order[0]]
                    category_rank = [str(classes[i]) for i in order]

                    row = {
                        "cluster": k,
                        "feature": feat,
                        "type": "categorical",
                        "mean_code": round(mean_code, round_to),
                        "most_likely_category": most_likely_cat,
                        "category_rank": category_rank,
                    }

                    rows.append(row)
                    continue   # skip numeric stats for categorical features

                # --------- NUMERIC FEATURES ----------
                mean_j = float(mu_orig[j])
                std_j  = float(std_orig[j])
                q_low  = mean_j + zL * std_j
                q_high = mean_j + zH * std_j

                row = {
                    "cluster": k,
                    "feature": feat,
                    "type": "numeric",
                    "q_low": round(q_low, round_to),
                    "q_high": round(q_high, round_to),
                    "mean": round(mean_j, round_to),
                    "std": round(std_j, round_to),
                }

                # optional: where the cluster mean sits in the global distribution (numeric only)
                if global_df is not None and feat in global_df.columns:
                    if pd.api.types.is_numeric_dtype(global_df[feat]):
                        gv = global_df[feat].to_numpy(dtype=float)
                        gv = gv[np.isfinite(gv)]
                        if gv.size > 0:
                            pct = float((gv <= mean_j).mean() * 100.0)
                            row["global_percentile"] = round(pct, 1)
                        else:
                            row["global_percentile"] = np.nan
                    else:
                        # non-numeric global column → no meaningful percentile
                        row["global_percentile"] = np.nan

                rows.append(row)

        return (
            pd.DataFrame(rows)
            .sort_values(["cluster", "feature"])
            .reset_index(drop=True)
        )
    

    def _rich_text_cluster_descriptions(
        self,
        desc_df: pd.DataFrame,
        cluster_prefix: str = "Cluster",
        threshold: float = 0.1,
        *,
        include_mean: bool = True,
        include_std: bool = True,
        include_global_percentile: bool = True,   # requires "global_percentile" column in desc_df
        round_to: int = 2,) -> Dict[int, str]:
        """
        Generate human-readable, bullet-list cluster descriptions.

        Expects desc_df to contain at least:
          ["cluster", "feature"]
        and for numeric rows:
          ["q_low", "q_high"]
        optionally:
          ["mean", "std", "global_percentile"]

        For categorical rows (type == "categorical"), expects:
          ["most_likely_category"] and optionally ["mean_code", "category_rank"].

        Returns
        -------
        dict: {cluster_id: multiline_string}
        """
        # Basic required columns
        needed = {"cluster", "feature"}
        missing = needed - set(desc_df.columns)
        if missing:
            raise ValueError(f"desc_df missing required columns: {missing}")

        has_type = "type" in desc_df.columns

        has_mean = "mean" in desc_df.columns and include_mean
        has_std = "std" in desc_df.columns and include_std
        has_pct = "global_percentile" in desc_df.columns and include_global_percentile

        # Set MultiIndex for easy lookup
        df = desc_df.set_index(["cluster", "feature"]).sort_index()

        # Respect user's feature order if provided; otherwise derive from df
        feature_order = list(getattr(self, "feature_cols", [])) or \
                        list(df.index.get_level_values("feature").unique())

        texts: Dict[int, str] = {}
        for k in sorted(df.index.get_level_values("cluster").unique()):
            lines = [f"{cluster_prefix} {int(k)} characteristics:"]
            any_feature_kept = False

            for feat in feature_order:
                if (k, feat) not in df.index:
                    continue

                row = df.loc[(k, feat)]

                row_type = row["type"] if has_type else "numeric"

                # ---------------- CATEGORICAL FEATURES ----------------
                if row_type == "categorical":
                    # We expect at least "most_likely_category"
                    if "most_likely_category" not in row.index:
                        continue  # nothing useful to say

                    cat = row["most_likely_category"]
                    segment = f"- {feat}: mostly '{cat}'"

                    extras = []

                    # Optionally show mean_code (cluster center in encoded space)
                    if "mean_code" in row.index:
                        extras.append(f"encoded_cluster_center {row['mean_code']:.{round_to}f}")

                    # Optionally show ranked categories
                    if "category_rank" in row.index:
                        try:
                            rank = list(row["category_rank"])
                        except Exception:
                            rank = None
                        if rank and len(rank) > 1:
                            # Show top 3 alternatives if available
                            others = [r for r in rank if r != cat][:3]
                            if others:
                                extras.append("other likely categories: " + ", ".join(map(str, others)))

                    if extras:
                        segment += " (" + ", ".join(extras) + ")"

                    lines.append(segment)
                    any_feature_kept = True
                    continue  # skip numeric logic for this feature

                # ---------------- NUMERIC FEATURES ----------------
                # For numeric rows we require q_low / q_high
                if not {"q_low", "q_high"}.issubset(row.index):
                    # If somehow missing, skip this numeric feature
                    continue

                lo, hi = row["q_low"], row["q_high"]

                # Skip if range too narrow
                if (hi - lo) <= threshold:
                    continue

                # Build the per-feature line
                segment = f"- {feat}: {lo:.{round_to}f} \u2192 {hi:.{round_to}f}"

                extras = []
                if has_mean and "mean" in row.index:
                    extras.append(f"cluster_mean {row['mean']:.{round_to}f}")
                if has_std and "std" in row.index:
                    extras.append(f"cluster_std {row['std']:.{round_to}f}")
                if has_pct and "global_percentile" in row.index and pd.notna(row["global_percentile"]):
                    # show as ~NNth percentile globally
                    gp = float(row["global_percentile"])
                    # Render 1 decimal if it’s not close to an integer
                    gp_str = f"{gp:.1f}" if abs(gp - round(gp)) >= 0.05 else f"{int(round(gp))}"
                    suffix = (
                        "st" if gp_str.endswith("1") and gp_str != "11" else
                        "nd" if gp_str.endswith("2") and gp_str != "12" else
                        "rd" if gp_str.endswith("3") and gp_str != "13" else
                        "th"
                    )
                    extras.append(f"~{gp_str}{suffix} percentile globally")

                if extras:
                    segment += " (" + ", ".join(extras) + ")"

                lines.append(segment)
                any_feature_kept = True

            if not any_feature_kept:
                lines.append("- (no strong feature variation)")

            texts[int(k)] = "\n".join(lines)

        return texts
    



#---------------------------------- Fuzzy C-Means Clusterer --------------------------------------    
class FuzzyCMeans_clusterer(Clusterer_template):
    def __init__(self):
        self.centers = None
        self.u = None              # membership matrix [n_samples, n_clusters]
        self.scaler = None
        self.feature_cols = None
        self.m = 2.0
        self.max_iter = 300
        self.error = 1e-5
        self.random_state = 42
        self.best_k = None

    def Create_clusters(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        K_max: int = 11,
        *,
        classification_encoder=None,
        scale: bool = True,
        m: float = 2.0,
        max_iter: int = 300,
        error: float = 1e-5,
        random_state: int = 42,
        prob_threshold: Optional[float] = None,
        cumprob_threshold: float = 0.8,
        max_experts: int = 3,
        Prob_cols: bool = False,
        hard_label_col: str = "Cluster_id",
        soft_prefix: str = "P_cluster_",
    ) -> pd.DataFrame:
        """
        Perform Fuzzy C-Means clustering with automatic K selection (via Xie–Beni index).
        Adds:
          - Hard labels
          - Soft membership columns
          - Overlapping 'Experts' list
        """
        import skfuzzy as fuzz

        self.feature_cols = feature_cols
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

        # --- Standardize features if requested ---
        if (classification_encoder!=None):
            df_unencoded=df.copy()
            df_encoded=transform_cat_encoder(df_unencoded, classification_encoder)
            
            X = df_encoded[feature_cols].to_numpy()
        else:
            X = df[feature_cols].to_numpy()
            
        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        X_T = X.T  # skfuzzy expects shape (features, samples)

        # --- (B) Model selection: sweep K and pick best by XB index ---
        best_score = np.inf
        best_cntr, best_u, best_k = None, None, None

        for k in range(2, K_max + 1):
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                X_T, c=k, m=m, error=error, maxiter=max_iter, seed=random_state
            )
            xb = self._xie_beni_index(X, cntr, u, m)
            if xb < best_score:
                best_score = xb
                best_cntr, best_u, best_k = cntr, u, k

        self.centers = best_cntr
        self.u = best_u.T   # transpose → (n_samples, K)
        self.best_k = best_k

        # --- (C) Hard + soft labels ---
        hard_labels = self.u.argmax(axis=1)
        output_df = df.copy()
        output_df[hard_label_col] = hard_labels

        if Prob_cols:
            for j in range(self.u.shape[1]):
                output_df[f"{soft_prefix}{j}"] = self.u[:, j]

        # --- (D) Overlapping experts ---
        experts_col = []
        for row in self.u:
            ranked = np.argsort(row)[::-1]
            selected = []

            if prob_threshold is not None:
                selected = [int(j) for j, p in enumerate(row) if p >= prob_threshold]
                if not selected:
                    cum, tmp = 0.0, []
                    for j in ranked:
                        tmp.append(int(j))
                        cum += row[j]
                        if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                            break
                    selected = tmp
            else:
                cum, tmp = 0.0, []
                for j in ranked:
                    tmp.append(int(j))
                    cum += row[j]
                    if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                        break
                selected = tmp

            selected = selected[:max_experts]
            if prob_threshold is not None and len(selected) == 0:
                selected = [-1]

            experts_col.append(selected)

        output_df["Experts"] = experts_col
        return output_df

    



    def Describe_clusters(self, clustered_df) -> Dict[int, str]:
        cluster_info_df = self._cluster_info(clustered_df)
        Cluster_description_dict = self._rich_text_cluster_descriptions(cluster_info_df)
        return Cluster_description_dict

    
    
    def _cluster_info(
        self,
        global_df: pd.DataFrame,                       # full dataset in ORIGINAL feature units
        quantiles: Tuple[float, float] = (0.1, 0.9),
        round_to: int = 3,
        *,
        use_membership_power: bool = True,             # use u^m as weights (standard in FCM)
        include_global_percentile: bool = True,) -> pd.DataFrame:
        """
        Fuzzy C-Means: membership-weighted per-cluster, per-feature stats, plus (optionally)
        the global percentile of the cluster's weighted mean.

        Requires:
          - self.u: (N, K) membership matrix (in [0,1])
          - self.centers: (K, D) cluster centers (not used for stats; we weight raw data)
          - self.feature_cols: list[str] of features
          - self.m or self.fuzziness (optional): fuzzifier (default=2.0 if absent)

        Returns a DataFrame with columns:
          ['cluster','feature','q_low','q_high','mean','std'] and
          optionally ['global_percentile'] if include_global_percentile=True
        """
        if not hasattr(self, "u") or self.u is None:
            raise RuntimeError("Fuzzy memberships self.u not found.")
        if not hasattr(self, "feature_cols"):
            raise RuntimeError("self.feature_cols not set.")

        u = np.asarray(self.u, dtype=float)          # (N, K)
        N, K = u.shape
        feats = list(self.feature_cols)
        q_lo, q_hi = quantiles

        # standard FCM uses weights w = u^m
        if use_membership_power:
            m = float(getattr(self, "m", getattr(self, "fuzziness", 2.0)))
            w_all = np.power(u, m)
        else:
            w_all = u

        rows = []
        for k in range(K):
            w = w_all[:, k]                           # (N,)
            w_sum = float(np.nansum(w))
            for feat in feats:
                x = global_df[feat].to_numpy(dtype=float)

                # Handle missing/inf
                mask = np.isfinite(x) & np.isfinite(w)
                x_m = x[mask]
                w_m = w[mask]
                if x_m.size == 0 or w_m.sum() <= 0:
                    row = {
                        "cluster": k, "feature": feat,
                        "q_low": np.nan, "q_high": np.nan,
                        "mean": np.nan, "std": np.nan
                    }
                    if include_global_percentile:
                        row["global_percentile"] = np.nan
                    rows.append(row)
                    continue

                # Weighted mean / std
                mu = float(np.average(x_m, weights=w_m))
                var = float(np.average((x_m - mu) ** 2, weights=w_m))
                sd  = float(np.sqrt(max(var, 0.0)))

                # Weighted quantiles within cluster
                q_vals =self. _weighted_quantiles(x_m, w_m, np.array([q_lo, q_hi], dtype=float))
                ql, qh = map(float, q_vals)

                row = {
                    "cluster": k,
                    "feature": feat,
                    "q_low": round(ql, round_to),
                    "q_high": round(qh, round_to),
                    "mean": round(mu, round_to),
                    "std": round(sd, round_to),
                }

                # Where does the cluster *mean* sit in the GLOBAL (unweighted) distribution?
                if include_global_percentile:
                    gv = x[np.isfinite(x)]
                    if gv.size > 0:
                        pct = float((gv <= mu).mean() * 100.0)
                        row["global_percentile"] = round(pct, 1)
                    else:
                        row["global_percentile"] = np.nan

                rows.append(row)

        return (
            pd.DataFrame(rows)
            .sort_values(["cluster", "feature"])
            .reset_index(drop=True)
        )
    
    def Find_experts(
            self,
            sample: Union[pd.Series, dict, np.ndarray],
            feature_cols: List[str],
            prob_threshold: Optional[float] = None,
            cumprob_threshold: float = 0.8,
            max_experts: int = 3,
            *,
            reference_df: Optional[pd.DataFrame] = None,   # full dataset (original units) for quantiles
            return_feature_quantiles: bool = True,) -> Dict[str, Any]:
            """
            Assign a new sample to fuzzy clusters (FCM) and compute in-cluster quantiles.
    
            Returns
            -------
            {
              'Cluster_id': int,                 # argmax membership
              'probs': np.ndarray,               # shape (K,), normalized memberships for sample
              'Experts': List[int],              # selected clusters per thresholds
              'Quantile_in_cluster': float,      # radius-based percentile within assigned cluster (0..1)
              'Radius_quantile': float,          # same as above (alias)
              'Per_feature_quantiles': dict[str,float]  # optional per-feature quantiles in assigned cluster
            }
            """
            # --- (A) Build sample vector in original + scaled spaces ---
            if isinstance(sample, (pd.Series, dict)):
                x_orig = np.array([sample[c] for c in feature_cols], dtype=float)
            else:
                x_orig = np.asarray(sample, dtype=float)
    
            # Keep a scaled copy for distance/membership computation
            if getattr(self, "scaler", None) is not None:
                x_scaled = self.scaler.transform(x_orig.reshape(1, -1))[0]
            else:
                x_scaled = x_orig.copy()
    
            # --- (B) Compute fuzzy memberships for the sample ---
            # Distances to cluster centers (centers assumed in scaled space)
            dists = la.norm(x_scaled - self.centers, axis=1)
            dists = np.where(dists == 0, 1e-12, dists)
    
            m = float(getattr(self, "m", getattr(self, "fuzziness", 2.0)))
            power = 2.0 / (m - 1.0)
            denom = (1.0 / dists) ** power
            u_new = denom / denom.sum()  # normalized memberships for the sample
            hard_label = int(np.argmax(u_new))
    
            # --- (C) Select overlapping experts as before ---
            ranked = np.argsort(u_new)[::-1]
            if prob_threshold is not None:
                selected = [int(j) for j, p in enumerate(u_new) if p >= prob_threshold]
                if not selected:
                    cum, tmp = 0.0, []
                    for j in ranked:
                        tmp.append(int(j))
                        cum += u_new[j]
                        if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                            break
                    selected = tmp
            else:
                cum, tmp = 0.0, []
                for j in ranked:
                    tmp.append(int(j))
                    cum += u_new[j]
                    if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                        break
                selected = tmp
            selected = selected[:max_experts]
    
            # --- (D) In-cluster quantiles (radius + per-feature), using reference data ---
            q_radius = np.nan
            per_feat_q: Optional[Dict[str, float]] = None
    
            if reference_df is not None:
                # Membership weights for training data, using u^m (standard in FCM stats)
                U = np.asarray(self.u, dtype=float)                 # (N, K)
                Wm = np.power(U, m)                                 # (N, K)
                w_k = Wm[:, hard_label]                             # weights for assigned cluster
    
                # Build training matrix in the *same scaled space* used by FCM
                X_train_orig = reference_df[feature_cols].to_numpy(dtype=float)
                if getattr(self, "scaler", None) is not None:
                    X_train_scaled = self.scaler.transform(X_train_orig)
                else:
                    X_train_scaled = X_train_orig
    
                # ---- Radius-based quantile (recommended scalar "typicality" score) ----
                center_k = self.centers[hard_label]                 # scaled center
                radii = la.norm(X_train_scaled - center_k, axis=1)  # (N,)
                r_star = float(la.norm(x_scaled - center_k))
                q_radius = self._weighted_percentile_of_value(radii, w_k, r_star)
    
                # ---- Per-feature quantiles (optional, human-insightful) ----
                if return_feature_quantiles:
                    per_feat_q = {}
                    # Use original units for per-feature readability
                    x_feat_orig = x_orig
                    for j, feat in enumerate(feature_cols):
                        col = reference_df[feat].to_numpy(dtype=float)
                        qj = self._weighted_percentile_of_value(col, w_k, float(x_feat_orig[j]))
                        per_feat_q[feat] = qj
    
            return {
                "Cluster_id": hard_label,
                "probs": u_new,
                "Experts": selected,
                "Quantile_in_cluster": q_radius,   # scalar 0..1 (radius-based)
                "Radius_quantile": q_radius,       # alias for clarity
                "Per_feature_quantiles": per_feat_q,
            }





    def _rich_text_cluster_descriptions(
            self,
            desc_df: pd.DataFrame,
            cluster_prefix: str = "Cluster",
            threshold: float = 0.1,
            *,
            include_mean: bool = True,
            include_std: bool = True,
            include_global_percentile: bool = True,   # requires "global_percentile" column in desc_df
            round_to: int = 2,) -> Dict[int, str]:
            """
            Generate human-readable, bullet-list cluster descriptions.
        
            Expects desc_df to contain at least:
              ["cluster", "feature", "q_low", "q_high"]
            and optionally:
              ["mean", "std", "global_percentile"]
        
            Returns
            -------
            dict: {cluster_id: multiline_string}
            """
            # Ensure sorting and easy lookup
            needed = {"cluster", "feature", "q_low", "q_high"}
            missing = needed - set(desc_df.columns)
            if missing:
                raise ValueError(f"desc_df missing required columns: {missing}")
        
            has_mean = "mean" in desc_df.columns and include_mean
            has_std = "std" in desc_df.columns and include_std
            has_pct = "global_percentile" in desc_df.columns and include_global_percentile
        
            df = desc_df.set_index(["cluster", "feature"]).sort_index()
        
            # Respect user's feature order if provided; otherwise derive from df
            feature_order = list(getattr(self, "feature_cols", [])) or \
                            list(df.index.get_level_values("feature").unique())
        
            texts: Dict[int, str] = {}
            for k in sorted(df.index.get_level_values("cluster").unique()):
                lines = [f"{cluster_prefix} {int(k)} characteristics:"]
                any_feature_kept = False
        
                for feat in feature_order:
                    if (k, feat) not in df.index:
                        continue
                    row = df.loc[(k, feat)]
                    lo, hi = row["q_low"], row["q_high"]
        
                    # Skip if range too narrow
                    if (hi - lo) <= threshold:
                        continue
        
                    # Build the per-feature line
                    segment = f"- {feat}: {lo:.{round_to}f} \u2192 {hi:.{round_to}f}"
        
                    extras = []
                    if has_mean:
                        extras.append(f"cluster_mean {row['mean']:.{round_to}f}")
                    if has_std:
                        extras.append(f"cluster_std {row['std']:.{round_to}f}")
                    if has_pct:
                        # show as ~NNth percentile globally
                        gp = float(row["global_percentile"])
                        # Render 1 decimal if it’s not close to an integer
                        gp_str = f"{gp:.1f}" if abs(gp - round(gp)) >= 0.05 else f"{int(round(gp))}"
                        suffix = "st" if gp_str.endswith("1") and gp_str != "11" else \
                                 "nd" if gp_str.endswith("2") and gp_str != "12" else \
                                 "rd" if gp_str.endswith("3") and gp_str != "13" else "th"
                        extras.append(f"~{gp_str}{suffix} percentile globally")
        
                    if extras:
                        segment += " (" + ", ".join(extras) + ")"
        
                    lines.append(segment)
                    any_feature_kept = True
        
                if not any_feature_kept:
                    lines.append("- (no strong feature variation)")
        
                texts[int(k)] = "\n".join(lines)
        
            return texts
    
    def _weighted_percentile_of_value(self, x: np.ndarray, w: np.ndarray, v: float) -> float:
        """
        Return the weighted ECDF value P(X <= v) in [0,1].
        """
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(x) & np.isfinite(w)
        x, w = x[mask], w[mask]
        if x.size == 0 or w.sum() <= 0:
            return np.nan
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]
        cdf = np.cumsum(w_sorted)
        cdf /= cdf[-1]
        # interpolate ECDF at v
        return float(np.interp(v, x_sorted, cdf, left=0.0, right=1.0))
    
    def _weighted_quantiles(self, x: np.ndarray, w: np.ndarray, qs: np.ndarray) -> np.ndarray:
        """
        Weighted quantiles of x for quantile levels qs in [0,1].
        """
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        qs = np.asarray(qs, dtype=float)
    
        mask = np.isfinite(x) & np.isfinite(w)
        x, w = x[mask], w[mask]
        if x.size == 0 or w.sum() <= 0:
            return np.full_like(qs, np.nan, dtype=float)
    
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = w[order]
        cdf = np.cumsum(w_sorted) / w_sorted.sum()
        return np.interp(qs, cdf, x_sorted)

    def _xie_beni_index(self, X: np.ndarray, centers: np.ndarray, u: np.ndarray, m: float) -> float:
        """
        Compute Xie–Beni index for fuzzy clustering.
        Lower XB = better clustering.
        """
        # Ensure u is (n_samples, n_clusters)
        if u.shape[0] == centers.shape[0]:
            u = u.T

        n_samples, _ = X.shape
        n_clusters = centers.shape[0]

        # Distances: (n_samples, n_clusters)
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

        # Membership-weighted dispersion
        num = np.sum((u ** m) * (dists ** 2))

        # Minimum distance between cluster centers
        dmin = np.min([
            np.linalg.norm(centers[i] - centers[j])
            for i in range(n_clusters) for j in range(n_clusters) if i != j
        ])

        xb = num / (n_samples * (dmin ** 2 + 1e-12))
        return xb
