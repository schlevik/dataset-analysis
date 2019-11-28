import logging
from typing import Tuple, List

import numpy as np
import sklearn

from util import tqdm

from metrics import Scorer, prepare
from util import Sample, Entity


def extract_features(q, s, rest, scorers):
    return [scorer(q, s, rest) for scorer in scorers]


logger = logging.getLogger("learn")
logger.setLevel("INFO")


def data_from_entry(sample, annotation, scorers,
                    extract_with) -> Tuple[List[List[int]], List[int]]:
    q, sfs, sentences = prepare(sample, annotation, extract_with,
                                just_supp_fact_ids=True)
    features = []
    labels = []
    for s_id, s in enumerate(sentences):
        rest = sentences[:]
        rest.remove(s)

        # TODO: for HotpotQA this needs to change
        features.append(extract_features(q, s, rest, scorers) + [s_id])
        labels.append(1 if s_id in sfs else 0)
    return features, labels


def do_balance(features, labels, shuffle=False):
    num_pos = sum(labels)
    num_neg = sum(abs(1 - l) for l in labels)
    max_size = num_pos if num_pos < num_neg else num_neg
    features_pos = []
    features_neg = []
    labels_pos = []
    labels_neg = []
    for f, l in zip(features, labels):
        if l == 0:
            features_neg.append(f)
            labels_neg.append(l)
        if l == 1:
            features_pos.append(f)
            labels_pos.append(l)
    indices_neg = np.random.choice(len(features_neg), max_size, replace=False)
    indices_pos = np.random.choice(len(features_pos), max_size, replace=False)
    features = [features_pos[i] for i in indices_pos] + [features_neg[i] for i
                                                         in indices_neg]
    labels = [labels_pos[i] for i in indices_pos] + [labels_neg[i] for i in
                                                     indices_neg]
    # if shuffle:
    #    np.random.shuffle(features)
    #    np.random.shuffle(labels)
    return features, labels


def flat_dataset_from_sample(samples: List[Tuple[Sample, List[Entity]]],
                             scorers: List[Scorer], paragraph_extract,
                             balance=False):
    features = []
    labels = []
    for i, (sample, annotation) in enumerate(tqdm(samples)):
        try:
            f, l = data_from_entry(sample, annotation, scorers,
                                   paragraph_extract)
        except ValueError as e:
            raise ValueError(f"{i}: {str(e)}")
        features.extend(f)
        labels.extend(l)
    if balance:
        features, labels = do_balance(features, labels)

    return np.array(features), np.array(labels)


def dataset_from_sample(samples: List[Tuple[Sample, List[Entity]]],
                        scorers: List[Scorer], paragraph_extract):
    docs = []
    for i, (sample, annotation) in enumerate(tqdm(samples)):
        features, labels = data_from_entry(sample, annotation, scorers,
                                           paragraph_extract)
        docs.append((features, labels))
    return docs


def f1_supporting_facts(predictions, ground_truth):
    predictions = [int(p) for p in predictions]
    tp = sum(p + g == 2 for p, g in zip(predictions, ground_truth))
    try:
        precision = tp / sum(
            predictions)  # tp / tp + fp (all the predicted ones)
    except ZeroDivisionError:
        precision = 0
    recall = tp / sum(ground_truth)  # tp / tp + fn (all the gold ones)
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        if precision + recall == 0:
            f1 = 0
        else:
            raise ZeroDivisionError
    return precision, recall, f1


def flatten(list_of_lists):
    features = []
    labels = []
    for document in list_of_lists:
        for sentence_features, sentence_labels in document:
            features.extend(sentence_features)
            labels.extend(sentence_labels)
    return features, labels


def cv_on_dataset_with_f1_on_positives(dataset, get_classifier, k=0,
                                       balance_train=True, max_n=0):
    # mean score for whole dataset
    pruned_ds = []
    for features, labels in dataset:
        if sum(labels) > 0:
            pruned_ds.append((features, labels))

    logger.info(f"{len(pruned_ds)} examples.")

    if 0 < max_n < len(pruned_ds):
        logger.info(f"removing {len(pruned_ds) - max_n} samples.")
        indices = np.random.choice(len(dataset), max_n, replace=False)
        pruned_ds = [pruned_ds[i] for i in indices]
        assert len(pruned_ds) == max_n
    mp = []
    mr = []
    mf1 = []
    np.random.shuffle(pruned_ds)
    if k == 0:
        k = len(pruned_ds)
    buckets = [pruned_ds[i::k] for i in range(k)]
    for i, bucket in enumerate(tqdm(buckets)):

        new_buckets = buckets[:]
        new_buckets.remove(bucket)
        features, labels = flatten(new_buckets)
        if balance_train:
            features, labels = do_balance(features, labels)

        features = np.array(features)
        labels = np.array(labels)
        classifier = get_classifier()
        classifier.fit(features, labels)
        # average per bucket
        ap = []
        ar = []
        af1 = []
        for X, y in bucket:
            predictions = classifier.predict(np.array(X))
            # actual per document
            p, r, f1 = f1_supporting_facts(predictions, y)
            ap.append(p)
            ar.append(r)
            af1.append(f1)
        ap = sum(ap) / len(ap)
        ar = sum(ar) / len(ar)
        af1 = sum(af1) / len(af1)
        logger.debug(f"Average score for iteration {i}: "
                     f"P: {ap:2f} R: {ar:2f}, F1: {af1:2f}")
        mp.append(ap)
        mr.append(ar)
        mf1.append(af1)

    return sum(mp) / len(mp), sum(mr) / len(mr), sum(mf1) / len(mf1)


def cv_the_cv(dataset, classifier, k, n, balance_train=True, max_n=0):
    results = [
        cv_on_dataset_with_f1_on_positives(dataset, classifier, k,
                                           balance_train, max_n=max_n)
        for _ in range(n)]
    results = np.array(results)
    p = results[:, 0]
    r = results[:, 1]
    f1 = results[:, 2]
    logger.info(f"P: {p.mean():0.2f} (+/- {p.std() * 2:0.2f})")
    logger.info(f"R: {r.mean():0.2f} (+/- {r.std() * 2:0.2f})")
    logger.info(f"F1: {f1.mean():0.2f} (+/- {f1.std() * 2:0.2f})")
    return results
