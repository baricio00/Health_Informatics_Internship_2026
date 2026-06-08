"""
Selects patients with QC RESULT 2 or 3 that are also in cv_splits.csv,
then randomly reassigns balanced folds and writes cv_splits_qc.csv.
"""

import csv
import random

SEED = 42
QC_FILE = "data/qc_results.csv"
CV_FILE = "data/cv_splits.csv"
OUT_FILE = "data/cv_splits_qc.csv"
N_FOLDS = 6  # folds: -1 (test), 0-4 (train)


def main():
    # Patients with QC RESULT 2 or 3
    qc_patients = set()
    with open(QC_FILE) as f:
        for row in csv.DictReader(f):
            if row["QC RESULT"].strip() in {"2", "3"}:
                qc_patients.add(row["PATIENT"].strip())

    print(f"Patients with QC RESULT 2 or 3: {len(qc_patients)}")

    # Patients present in cv_splits
    cv_patients = []
    with open(CV_FILE) as f:
        for row in csv.DictReader(f):
            cv_patients.append(row["patient_id"].strip())

    cv_set = set(cv_patients)

    # Intersection
    selected = sorted(qc_patients & cv_set)
    print(f"Patients also in cv_splits:      {len(cv_set)}")
    print(f"Intersection (final subset):     {len(selected)}")

    # Randomly reassign balanced folds
    random.seed(SEED)
    shuffled = selected[:]
    random.shuffle(shuffled)

    # Distribute as evenly as possible: first (n % N_FOLDS) folds get one extra patient
    n = len(shuffled)
    base, extra = divmod(n, N_FOLDS)
    fold_sizes = [base + (1 if i < extra else 0) for i in range(N_FOLDS)]

    # Fold labels: -1 (validation), 0-4 (train)
    fold_labels = list(range(-1, N_FOLDS - 1))  # [-1, 0, 1, 2, 3, 4]

    assignments = []
    idx = 0
    for label, size in zip(fold_labels, fold_sizes):
        for _ in range(size):
            assignments.append((shuffled[idx], label))
            idx += 1

    print(f"\nFold distribution:")
    from collections import Counter
    print(Counter(fold for _, fold in assignments))

    # Write output
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "fold"])
        for patient_id, fold in sorted(assignments):
            writer.writerow([patient_id, fold])

    print(f"\nSaved {len(assignments)} patients to {OUT_FILE}")


if __name__ == "__main__":
    main()
