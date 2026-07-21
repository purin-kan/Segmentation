# Metrics

| Metric | Unit | Range | Best | Level | Averaging |
| - | - | - | - | - | - |
| Dice | ratio | 0 to 1 | 1.0 | Region | per layer |
| IoU | ratio | 0 to 1 | 1.0 | Region | per layer |
| MAD | px (~3.87 µm/px axial) | 0 to ∞ | 0 | Boundary | per boundary |
| RMSE | px | 0 to ∞ | 0 | Boundary | per boundary |
| Coverage | ratio | 0 to 1 | 1.0 | Boundary | per boundary |

## Notes

- **Dice** — `2 × overlap / (area_pred + area_true)`
  - Inflates for thick layers regardless of boundary accuracy.
  - Example: the composite ONL→BM layer (layer 5). Its Dice is not merely inflated but
    uninformative about either of its own boundaries: the thick middle dominates the overlap,
    so error at OPL/ONL or BM barely moves the score.
  - Supporting metric, not the headline.
- **IoU** — `overlap / union`
  - Always ≤ Dice for the same two masks.
  - Stricter of the two.
- **MAD** — mean `|y_true - y_pred|` per column
  - Unaffected by layer thickness, unlike Dice.
  - Headline metric for this experiment.
  - Scored only over columns both annotated and predicted.
  - See Coverage for the unscored rest.
- **RMSE** — sqrt(mean `(y_true - y_pred)^2`) per column
  - Always ≥ MAD for the same data.
  - Squaring weights large individual misses more than MAD.
  - Similar MAD but higher RMSE than another method means a few large errors, not a uniform offset.
- **Coverage** — fraction of annotated columns predicted
  - Not an accuracy metric.
  - Measures whether the method attempted a prediction, not whether it was correct.
  - Ordinal-identity methods (1a, 2, 4) are expected to score low coverage and high MAD on low-contrast inner boundaries.

## Reading them together

- Check Coverage alongside MAD/RMSE: a method that skips hard columns can look artificially accurate on MAD/RMSE alone.
- Check MAD alongside Dice: a thick composite layer can mask real boundary error in Dice.
