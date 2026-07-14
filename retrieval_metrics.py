"""
Retrieval metrics from scratch: Recall@k and Precision@k.

These are the classic information-retrieval metrics that underpin RAGAS-style
retrieval scoring. Given a set of relevant ("actual") items and a ranked list of
"predicted" items, they answer two complementary questions at a cut-off ``k``:

  - Recall@k    -- of ALL the relevant items, how many appear in the top k?
  - Precision@k -- of the top k returned items, how many are actually relevant?

Recall rewards finding everything; precision rewards not returning junk. Watching
both change as k grows builds intuition for the trade-off between them.

How to run
----------
    python retrieval_metrics.py

Pure Python -- no dependencies to install.
"""


def recall(actual, predicted, k):
    """Fraction of the relevant items that appear within the top ``k`` predictions."""
    act_set = set(actual)
    pred_set = set(predicted[:k])           # only the top-k predictions count
    intersection_set = act_set & pred_set   # relevant items that were retrieved
    result = round(len(intersection_set) / float(len(act_set)), 2)
    return result


def precision(actual, predicted, k):
    """Fraction of the top ``k`` predictions that are actually relevant."""
    act_set = set(actual)
    pred_set = set(predicted[:k])
    intersection_set = act_set & pred_set
    result = round(len(intersection_set) / float(len(pred_set)), 2)
    return result


# Worked example: 4 relevant items hidden inside a ranked list of 8 predictions.
actual = ["2", "4", "5", "7"]
predicted = ["1", "2", "3", "4", "5", "6", "7", "8"]

print(f"Actual relevant items: {actual}")
print(f"Predicted ranking:     {predicted}\n")

# Recall climbs (or holds) as k grows -- a bigger window can only find more.
print("Recall@k  (fraction of relevant items found within the top k):")
for k in range(4, 9):
    print(f"  Recall@{k} = {recall(actual, predicted, k)}")

# Precision tends to fall as k grows -- later, less-relevant items dilute the top k.
print("\nPrecision@k  (fraction of the top k that are relevant):")
for k in range(1, 9):
    print(f"  Precision@{k} = {precision(actual, predicted, k)}")
