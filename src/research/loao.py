"""Leave-One-Attack-Out split helpers for robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LOAOSplit:
    """Container for one leave-one-attack-out split."""

    held_out_attack: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class LeaveOneAttackOut:
    """Generates LOAO splits based on attack label values."""

    def __init__(
        self,
        label_column: str = "label",
        benign_label: str = "benign",
        benign_test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self.label_column = label_column
        self.benign_label = benign_label
        self.benign_test_size = benign_test_size
        self.random_state = random_state

    def attack_labels(self, frame: pd.DataFrame) -> List[str]:
        """Returns sorted attack labels excluding benign."""
        unique_labels = sorted(set(frame[self.label_column].astype(str)))
        return [label for label in unique_labels if label != self.benign_label]

    def split(self, frame: pd.DataFrame) -> Iterator[LOAOSplit]:
        """Yields one split per held-out attack label.

        Each split uses non-overlapping benign subsets for train and test to
        prevent leakage between LOAO train and evaluation sets.
        """
        attacks = self.attack_labels(frame)
        benign_df = frame.loc[frame[self.label_column] == self.benign_label]

        if benign_df.empty:
            raise ValueError("LOAO requires benign samples but none were found")
        if len(benign_df) < 2:
            raise ValueError("LOAO requires at least two benign samples")

        for split_index, held_out in enumerate(attacks):
            held_out_df = frame.loc[frame[self.label_column] == held_out]
            remaining_attacks_df = frame.loc[
                (frame[self.label_column] != self.benign_label)
                & (frame[self.label_column] != held_out)
            ]

            if held_out_df.empty:
                continue

            benign_train_df, benign_test_df = train_test_split(
                benign_df,
                test_size=self.benign_test_size,
                random_state=self.random_state + split_index,
                shuffle=True,
            )

            train_df = pd.concat([benign_train_df, remaining_attacks_df], axis=0).sample(
                frac=1.0,
                random_state=self.random_state,
            )
            test_df = pd.concat([benign_test_df, held_out_df], axis=0).sample(
                frac=1.0,
                random_state=self.random_state,
            )
            yield LOAOSplit(
                held_out_attack=held_out,
                train_df=train_df.reset_index(drop=True),
                test_df=test_df.reset_index(drop=True),
            )

    def summarize(self, frame: pd.DataFrame) -> Dict[str, int]:
        """Summarizes row count per attack label."""
        counts = frame[self.label_column].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}
