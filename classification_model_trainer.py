</> Python

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


class ClassificationTrainer:
    def __init__(self, output_dir="outputs/classification"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = None

    def create_dataset(self):
        X, y = make_classification(
            n_samples=1200,
            n_features=14,
            n_informative=8,
            n_redundant=3,
            n_classes=2,
            weights=[0.6, 0.4],
            class_sep=1.3,
            random_state=42
        )

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        return X, y

    def build_model(self):
        rf = RandomForestClassifier(random_state=42)
        gb = GradientBoostingClassifier(random_state=42)

        voting_model = VotingClassifier(
            estimators=[
                ("random_forest", rf),
                ("gradient_boosting", gb)
            ],
            voting="soft"
        )

        return voting_model

    def tune_model(self, X_train, y_train):
        model = self.build_model()

        param_grid = {
            "random_forest__n_estimators": [100, 200],
            "random_forest__max_depth": [5, 8, None],
            "gradient_boosting__n_estimators": [100, 150],
            "gradient_boosting__learning_rate": [0.03, 0.05]
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        self.best_model = search.best_estimator_

        print("최적 파라미터:", search.best_params_)
        print("최적 CV F1:", search.best_score_)

    def evaluate(self, X_test, y_test):
        pred = self.best_model.predict(X_test)
        prob = self.best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob)
        }

        report = classification_report(y_test, pred)

        pd.DataFrame([metrics]).to_csv(
            self.output_dir / "metrics.csv",
            index=False
        )

        with open(self.output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print("평가 결과")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print(report)

    def save_model(self):
        model_path = self.output_dir / "best_model.pkl"
        joblib.dump(self.best_model, model_path)
        print("모델 저장 완료:", model_path)

    def run(self):
        X, y = self.create_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.tune_model(X_train, y_train)
        self.evaluate(X_test, y_test)
        self.save_model()


if __name__ == "__main__":
    trainer = ClassificationTrainer()
    trainer.run()
