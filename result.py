from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Result:
    model_name: str
    dataset: str
    sparsing_name: str
    acc: Optional[list[float]] = field(default=None, compare=False)
    power: Optional[float] = field(default=None, compare=False)
    removed_percentage: Optional[float] = field(default=None, compare=False)

    def __str__(self) -> str:
        """Provides a formatted string representation of the result."""
        lines = [
            f"Model Name: {self.model_name}",
            f"Dataset: {self.dataset}",
            f"Sparsing Name: {self.sparsing_name}",
            f"Power: {self.power:.2g}" if self.power is not None else "Power: None",
        ]

        if self.acc:
            lines.extend([
                f"Accuracy Mean: {torch.tensor(self.acc).mean():.2%}",
                f"Accuracy Std: {torch.tensor(self.acc).std():.2%}",
            ])
        else:
            lines.append("Accuracy: N/A")

        if self.removed_percentage is not None:
            lines.append(f"Removed %: {self.removed_percentage:.2%}")
        else:
            lines.append("Removed %: N/A")
        lines.append("\n")

        return "\n".join(lines)

    def as_dict(self) -> dict[str, str]:
        acc_mean = f'{torch.tensor(self.acc).mean():.2%}' if self.acc else 'N/A'
        acc_std = f'{torch.tensor(self.acc).std():.2%}' if self.acc else 'N/A'

        return {
            'Model Name': self.model_name,
            'Dataset': self.dataset,
            'Sparsing Name': self.sparsing_name,
            'Power': f'{self.power:.2g}' if self.power is not None else 'None',
            'Accuracy Mean': acc_mean,
            'Accuracy Std': acc_std,
            'Removed %': f'{self.removed_percentage:.2%}' if self.removed_percentage is not None else 'N/A'
        }
