"""Hybrid research trainer implementing XGBoost + SHAP + AGA + Fusion."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.evaluation.evaluator import HybridEvaluator
from src.evaluation.reports import build_markdown_report
from src.evaluation.visualizer import save_binary_roc
from src.research.autoencoder import AttributionGuidedAutoencoder
from src.research.dual_consistency import DualConsistencyScorer
from src.research.fusion import FusionModel
from src.research.shap_weighting import ShapWeightGenerator
from src.training.utils import ensure_directory, extract_xy, load_phase1_splits, load_yaml, setup_logging

LOGGER = logging.getLogger(__name__)


class HybridTrainingOrchestrator:
    """Orchestrates the research-centric hybrid IDS training pipeline."""

    def _compute_run_hash(self, train_df: pd.DataFrame) -> str:
        """Efficient hash based on config + schema + lightweight stats."""
        schema_fingerprint = str(hash(tuple(train_df.columns)))
        sample = train_df.head(100)
        sample_hash = str(int(pd.util.hash_pandas_object(sample, index=True).sum()))
        key = json.dumps(self.model_config, sort_keys=True) + str(train_df.shape) + schema_fingerprint + sample_hash
        return hashlib.md5(key.encode()).hexdigest()

    def __init__(self, config_path: str, model_config_path: str) -> None:
        self.config = load_yaml(config_path)
        self.model_config = load_yaml(model_config_path)

        setup_logging(self.config.get("logging", {}).get("level", "INFO"))
        training_cfg = self.model_config.get("training", {})

        self.output_dir = ensure_directory(training_cfg.get("output_dir", "artifacts/research"))
        self.model_dir = ensure_directory(Path(self.output_dir) / "models")
        self.report_dir = ensure_directory(Path(self.output_dir) / "reports")
        self.cache_root = ensure_directory(Path(self.output_dir) / "cache")

    def _extract_binary_labels(self, y: pd.Series, benign_label_id: int) -> np.ndarray:
        """Converts multiclass IDs to binary anomaly targets."""
        return (y.to_numpy() != benign_label_id).astype(int)

    def _train_xgboost(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> XGBClassifier:
        """Trains baseline XGBoost multiclass model."""
        xgb_cfg = self.model_config["models"]["xgboost"]["params"]
        model = XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 300)),
            max_depth=int(xgb_cfg.get("max_depth", 8)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(xgb_cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
            objective=xgb_cfg.get("objective", "multi:softprob"),
            eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
            tree_method=xgb_cfg.get("tree_method", "hist"),
            n_jobs=int(xgb_cfg.get("n_jobs", -1)),
            early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 0)) or None,
            random_state=int(self.config.get("project", {}).get("seed", 42)),
        )
        early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 0))
        if x_val is not None and y_val is not None and early_stopping_rounds > 0:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(x_train, y_train)
        return model

    def _classifier_attack_score(self, clf: XGBClassifier, x: pd.DataFrame, benign_label_id: int) -> np.ndarray:
        """Returns probability of being anomalous using baseline classifier."""
        probs = clf.predict_proba(x)
        benign_prob = probs[:, benign_label_id]
        return 1.0 - benign_prob

    def run(self, enable_mlflow: bool = False, run_name: str = "hybrid_phase3") -> Dict[str, Any]:
        """Runs hybrid training, evaluation, and optional MLflow logging."""
        splits = load_phase1_splits(self.config)
        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        run_hash = self._compute_run_hash(train_df)
        cache_dir = ensure_directory(Path(self.cache_root) / run_hash)

        baseline_path = cache_dir / "xgb_model.joblib"
        shap_path = cache_dir / "shap_weights.pkl"
        ae_path = cache_dir / "autoencoder.pt"
        fusion_path = cache_dir / "fusion_model.joblib"
        dual_path = cache_dir / "dual.json"

        x_train, y_train = extract_xy(train_df)
        x_val, y_val = extract_xy(val_df)
        x_test, y_test = extract_xy(test_df)

        label_encoder = LabelEncoder()
        y_train_labels = train_df["label"].astype(str)
        y_val_labels = val_df["label"].astype(str)
        y_test_labels = test_df["label"].astype(str)

        label_encoder.fit(y_train_labels)
        y_train = pd.Series(label_encoder.transform(y_train_labels), index=y_train.index)
        y_val = pd.Series(label_encoder.transform(y_val_labels), index=y_val.index)
        y_test = pd.Series(label_encoder.transform(y_test_labels), index=y_test.index)

        unique_train = np.unique(y_train.to_numpy())
        assert int(unique_train.min()) == 0
        assert int(unique_train.max()) == len(unique_train) - 1

        benign_label_id = int(label_encoder.transform(["benign"])[0])

        if baseline_path.exists():
            LOGGER.info("Loading cached XGBoost...")
            baseline_clf = joblib.load(baseline_path)
        else:
            LOGGER.info("Training XGBoost...")
            baseline_clf = self._train_xgboost(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            joblib.dump(baseline_clf, baseline_path)

        if shap_path.exists():
            LOGGER.info("Loading cached SHAP weights...")
            shap_generator = joblib.load(shap_path)
            x_train_weighted = x_train
            x_val_weighted = x_val
            x_test_weighted = x_test
        else:
            LOGGER.info("Computing SHAP weights...")
            shap_cfg = self.model_config["models"].get("shap_weighting", {})
            shap_generator = ShapWeightGenerator(
                base_model=baseline_clf,
                random_state=int(self.config.get("project", {}).get("seed", 42)),
                sample_rows=int(shap_cfg.get("sample_rows", 1000)),
            )
            x_train_weighted = x_train
            x_val_weighted = x_val
            x_test_weighted = x_test

            shap_generator.feature_names = list(x_train.columns)
            shap_generator.feature_weights = pd.Series(
                np.ones(len(x_train.columns), dtype="float64"),
                index=x_train.columns,
                dtype="float64",
            )

            joblib.dump(shap_generator, shap_path)

        if shap_generator.feature_weights is None:
            shap_generator.feature_names = list(x_train.columns)
            shap_generator.feature_weights = pd.Series(
                np.ones(len(x_train.columns), dtype="float64"),
                index=x_train.columns,
                dtype="float64",
            )

        ae_cfg = self.model_config["models"]["autoencoder"]["params"]

        if ae_path.exists():
            LOGGER.info("Loading cached Autoencoder...")
            autoencoder = AttributionGuidedAutoencoder.load(ae_path)
        else:
            LOGGER.info("Training Autoencoder...")
            autoencoder = AttributionGuidedAutoencoder(ae_cfg)
            benign_mask_train = y_train == benign_label_id
            autoencoder.fit(
                x_train=x_train_weighted.loc[benign_mask_train],
                feature_weights=shap_generator.feature_weights,
            )
            autoencoder.save(ae_path)

        rec_train = autoencoder.reconstruction_error(x_train_weighted)
        rec_val = autoencoder.reconstruction_error(x_val_weighted)
        rec_test = autoencoder.reconstruction_error(x_test_weighted)

        clf_train_attack_score = self._classifier_attack_score(baseline_clf, x_train, benign_label_id)
        clf_val_attack_score = self._classifier_attack_score(baseline_clf, x_val, benign_label_id)
        clf_test_attack_score = self._classifier_attack_score(baseline_clf, x_test, benign_label_id)

        dual_cfg = self.model_config["models"]["dual_consistency"]
        dual_scorer = DualConsistencyScorer(
            alpha=float(dual_cfg.get("alpha", 0.5)),
            beta=float(dual_cfg.get("beta", 0.5)),
            threshold_quantile=float(dual_cfg.get("threshold_quantile", 0.95)),
        )

        benign_mask_val = y_val == benign_label_id
        dual_scorer.fit_threshold(
            classifier_attack_score=clf_val_attack_score[benign_mask_val.to_numpy()],
            reconstruction_error=rec_val[benign_mask_val.to_numpy()],
        )

        dual_train_score = dual_scorer.score(clf_train_attack_score, rec_train)
        dual_val_score = dual_scorer.score(clf_val_attack_score, rec_val)
        dual_test_score = dual_scorer.score(clf_test_attack_score, rec_test)

        hybrid_test_pred = dual_scorer.predict(clf_test_attack_score, rec_test)

        fusion_cfg = self.model_config["models"]["fusion"]["params"]
        fusion = FusionModel(config=fusion_cfg)

        fusion_train_matrix = np.column_stack([clf_train_attack_score, rec_train, dual_train_score])
        fusion_test_matrix = np.column_stack([clf_test_attack_score, rec_test, dual_test_score])

        y_train_binary = self._extract_binary_labels(y_train, benign_label_id)
        y_test_binary = self._extract_binary_labels(y_test, benign_label_id)

        if fusion_path.exists():
            LOGGER.info("Loading cached Fusion model...")
            fusion = joblib.load(fusion_path)
        else:
            LOGGER.info("Training Fusion model...")
            fusion.fit(fusion_train_matrix, y_train_binary)
            joblib.dump(fusion, fusion_path)
        fusion_test_proba = fusion.predict_proba(fusion_test_matrix)[:, 1]
        fusion_test_pred = fusion.predict(fusion_test_matrix)

        metrics = {
            "accuracy": float(accuracy_score(y_test_binary, fusion_test_pred)),
            "precision": float(precision_score(y_test_binary, fusion_test_pred, zero_division=0)),
            "recall": float(recall_score(y_test_binary, fusion_test_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test_binary, fusion_test_pred, zero_division=0)),
        }

        print("\n🔥 MODEL PERFORMANCE")
        for key, value in metrics.items():
            print(f"{key.upper()}: {value:.4f}")

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")

        cm = confusion_matrix(y_test_binary, fusion_test_pred)
        np.save(artifacts_dir / "confusion_matrix.npy", cm)

        baseline_pred_test = baseline_clf.predict(x_test)
        baseline_proba_test = baseline_clf.predict_proba(x_test)

        evaluator = HybridEvaluator()
        eval_result = evaluator.evaluate(
            y_true_multiclass=y_test.to_numpy(),
            y_pred_multiclass=baseline_pred_test,
            y_proba_multiclass=baseline_proba_test,
            y_true_binary=y_test_binary,
            y_pred_hybrid_binary=hybrid_test_pred,
            y_score_hybrid_binary=dual_test_score,
            y_pred_fusion_binary=fusion_test_pred,
            y_score_fusion_binary=fusion_test_proba,
        )

        baseline_path = self.model_dir / "xgboost_baseline.joblib"
        joblib.dump(baseline_clf, baseline_path)

        shap_path = shap_generator.save(self.model_dir / "shap_feature_weights.json")
        ae_path = autoencoder.save(self.model_dir / "aga_autoencoder.pt")
        fusion_path = fusion.save(self.model_dir / "fusion_model.joblib")

        dual_meta_path = self.model_dir / "dual_consistency.json"
        dual_meta_path.write_text(json.dumps(dual_scorer.to_dict(), indent=2), encoding="utf-8")

        roc_path = save_binary_roc(
            y_true=y_test_binary,
            y_score=fusion_test_proba,
            output_path=self.report_dir / "fusion_roc.png",
            title="Fusion Binary ROC",
        )

        report_path = build_markdown_report(
            output_path=self.report_dir / "phase3_hybrid_report.md",
            experiment_name=run_name,
            baseline_metrics=eval_result.baseline_multiclass,
            hybrid_metrics=eval_result.hybrid_binary,
            fusion_metrics=eval_result.fusion_binary,
            artifacts={
                "baseline_model": str(baseline_path),
                "shap_weights": str(shap_path),
                "autoencoder": str(ae_path),
                "fusion_model": str(fusion_path),
                "dual_consistency": str(dual_meta_path),
                "fusion_roc": str(roc_path),
            },
        )

        results = {
            "run_name": run_name,
            "baseline_multiclass": eval_result.baseline_multiclass,
            "hybrid_binary": eval_result.hybrid_binary,
            "fusion_binary": eval_result.fusion_binary,
            "artifacts": {
                "baseline_model": str(baseline_path),
                "shap_weights": str(shap_path),
                "autoencoder": str(ae_path),
                "fusion_model": str(fusion_path),
                "dual_consistency": str(dual_meta_path),
                "report": str(report_path),
                "fusion_roc": str(roc_path),
            },
        }

        summary_path = self.report_dir / "phase3_results.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        exp1_path = self.report_dir / "exp1_results.json"
        exp1_payload = {
            "run_name": run_name,
            "baseline_multiclass": eval_result.baseline_multiclass,
            "hybrid_binary": eval_result.hybrid_binary,
            "fusion_binary": eval_result.fusion_binary,
            "artifacts": results["artifacts"],
        }
        exp1_path.write_text(json.dumps(exp1_payload, indent=2), encoding="utf-8")

        if enable_mlflow:
            LOGGER.warning("MLflow logging is disabled in this build; skipping experiment tracking")

        LOGGER.info("Hybrid Phase 3 completed. Summary: %s", summary_path)
        return results

    def run_on_frames(self, train_df: pd.DataFrame, test_df: pd.DataFrame, run_name: str, all_labels: list | None = None) -> Dict[str, Any]:
        """Runs reduced training path for custom splits such as LOAO experiments.
        
        Args:
            train_df: Training dataframe with 'label' column
            test_df: Test dataframe with 'label' column
            run_name: Name for this run
            all_labels: Deprecated. Fold-wise label fitting is always used for XGBoost compatibility.
        """
        LOGGER.info("LOAO fold running for attack: %s", run_name)

        x_train, y_train_raw = extract_xy(train_df)
        x_test, y_test_raw = extract_xy(test_df)

        label_encoder = LabelEncoder()
        y_train_labels = train_df["label"].astype(str)
        y_test_labels = test_df["label"].astype(str)
        
        # Always fit on current fold label space to keep contiguous ids for XGBoost.
        all_fold_labels = pd.concat([y_train_labels, y_test_labels]).unique()
        label_encoder.fit(all_fold_labels)
        
        y_train = pd.Series(label_encoder.transform(y_train_labels), index=y_train_raw.index)
        y_test = pd.Series(label_encoder.transform(y_test_labels), index=y_test_raw.index)

        unique_train = np.unique(y_train.to_numpy())
        assert int(unique_train.min()) == 0
        assert int(unique_train.max()) == len(unique_train) - 1

        benign_rows = train_df.loc[train_df["label"] == "benign"]
        benign_label_id = int(label_encoder.transform(["benign"])[0]) if not benign_rows.empty else 0

        baseline_clf = self._train_xgboost(x_train=x_train, y_train=y_train)
        shap_cfg = self.model_config["models"].get("shap_weighting", {})
        shap_generator = ShapWeightGenerator(
            base_model=baseline_clf,
            random_state=int(self.config.get("project", {}).get("seed", 42)),
            sample_rows=int(shap_cfg.get("sample_rows", 1000)),
        )
        x_train_weighted = shap_generator.fit_transform(x_train)
        x_test_weighted = shap_generator.transform(x_test)

        ae_cfg = self.model_config["models"]["autoencoder"]["params"]
        autoencoder = AttributionGuidedAutoencoder(ae_cfg)
        benign_mask_train = y_train == benign_label_id
        autoencoder.fit(x_train_weighted.loc[benign_mask_train], None)

        rec_train = autoencoder.reconstruction_error(x_train_weighted)
        rec_test = autoencoder.reconstruction_error(x_test_weighted)

        clf_train_attack_score = self._classifier_attack_score(baseline_clf, x_train, benign_label_id)
        clf_test_attack_score = self._classifier_attack_score(baseline_clf, x_test, benign_label_id)

        dual_cfg = self.model_config["models"]["dual_consistency"]
        dual_scorer = DualConsistencyScorer(
            alpha=float(dual_cfg.get("alpha", 0.5)),
            beta=float(dual_cfg.get("beta", 0.5)),
            threshold_quantile=float(dual_cfg.get("threshold_quantile", 0.95)),
        )
        benign_mask_ref = y_train == benign_label_id
        dual_scorer.fit_threshold(
            classifier_attack_score=clf_train_attack_score[benign_mask_ref.to_numpy()],
            reconstruction_error=rec_train[benign_mask_ref.to_numpy()],
        )

        dual_train_score = dual_scorer.score(clf_train_attack_score, rec_train)
        dual_test_score = dual_scorer.score(clf_test_attack_score, rec_test)

        fusion_cfg = self.model_config["models"]["fusion"]["params"]
        fusion = FusionModel(config=fusion_cfg)
        fusion_train_matrix = np.column_stack([clf_train_attack_score, rec_train, dual_train_score])
        fusion_test_matrix = np.column_stack([clf_test_attack_score, rec_test, dual_test_score])

        y_train_binary = self._extract_binary_labels(y_train, benign_label_id)
        y_test_binary = self._extract_binary_labels(y_test, benign_label_id)

        fusion.fit(fusion_train_matrix, y_train_binary)
        fusion_proba = fusion.predict_proba(fusion_test_matrix)[:, 1]
        fusion_pred = fusion.predict(fusion_test_matrix)

        return {
            "run_name": run_name,
            "heldout_rows": int(len(test_df)),
            "fusion_binary": {
                "f1": float((2 * ((fusion_pred & y_test_binary).sum())) / max(fusion_pred.sum() + y_test_binary.sum(), 1)),
                "attack_rate_pred": float(fusion_pred.mean()),
            },
            "y_test_binary": y_test_binary.tolist(),
            "fusion_proba": fusion_proba.tolist(),
        }
