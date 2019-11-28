import string
from abc import ABC, abstractmethod
from math import log
from typing import List, Tuple, Generator

import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from util import tqdm

from util import Entity, Sample, get_question
from nltk import word_tokenize, SnowballStemmer, defaultdict
import ailog
import logging

ailog.setup_logging('logging.conf')
logger = logging.getLogger("This")

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')


class Scorer(ABC):
    def __init__(self, remove_punctuation=True, remove_stopwords=False,
                 do_stem=False):
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.do_stem = do_stem
        self.c = Curry()
        if self.remove_punctuation:
            self.c += unpunct
        if self.remove_stopwords:
            self.c += unstop
        if self.remove_punctuation:
            self.c += unpunct

    @abstractmethod
    def __call__(self, sentence, other_sentence, rest_sentences,
                 **kwargs) -> int:
        ...


class Curry:
    def __init__(self, function=None):
        if function:
            self.functions = [function]
        else:
            self.functions = []

    def __add__(self, other):
        c = Curry()
        c.functions = self.functions
        c.functions += [other]
        return c

    def __call__(self, arg):
        for f in self.functions:
            arg = f(arg)
        return arg


def fuzzy_substr(s1: str, s2: str):
    return s1 == s2


def remove_punctuation(s: str, trailing_only=False):
    if trailing_only:
        i = len(s) - 1
        while i > 0 and s[i] in string.punctuation:
            i -= 1
        return s[:i]
    return "".join(c for c in s if c not in string.punctuation)


def fuzzy_subsentence(sentence: List[str], other_sentence: List[str]):
    tries = len(other_sentence) - len(sentence) + 1
    if tries < 0:
        logger.debug(f'{sentence} in {other_sentence}: False')
        return False
    for i in range(tries):
        if all(fuzzy_substr(s1, s2) for s1, s2 in
               zip(sentence, other_sentence[i:])):
            logger.debug(f'{sentence} in {other_sentence}: True')
            return True
        logger.debug(f'{sentence} in {other_sentence}: False')
    return False


def stem(sentence: List[str]) -> Generator[str, None, None]:
    return (stemmer.stem(w) for w in sentence)


def unstop(sentence: List[str]) -> Generator[str, None, None]:
    return (w for w in sentence if w not in stop_words)


def unpunct(sentence: List[str]) -> Generator[str, None, None]:
    return (w for w in sentence if w not in string.punctuation)


def normalize(sentence: List[str]) -> List[str]:
    return [t.lower().strip() for t in sentence]


class MaxNgramScorer(Scorer):

    def __call__(self, sentence: List[str], other_sentence: List[str], *args,
                 **kwargs):

        if self.c.functions:
            sentence = list(self.c(sentence))
            other_sentence = list(self.c(other_sentence))

        max_ngram = 0
        sentence_idx = 0
        for i, _ in enumerate(sentence):
            subsentence = sentence[i:]
            for j, _ in enumerate(subsentence, 1):
                if fuzzy_subsentence(subsentence[:j],
                                     other_sentence) and j > max_ngram:
                    max_ngram = len(subsentence[:j])
                    sentence_idx = i
        return max_ngram


class GloveDistanceScorer(Scorer):
    def __init__(self, glove_file: str, m):
        super().__init__(do_stem=False)
        self.m = m
        self.words = dict()
        with open(glove_file, 'r') as f:
            content = f.readlines()
        for line in tqdm(content):
            word, embedding = line.split(" ", 1)
            embedding = np.array([float(val) for val in embedding.split(" ")])
            self.words[word] = embedding

    def w2v(self, word):
        try:
            return self.words[word].reshape(1, -1)
        except KeyError:
            return np.array([0] * 300, dtype=float).reshape(1, -1)

    def __call__(self, sentence: List[str], other_sentence: List[str], *args,
                 **kwargs):
        if self.c.functions:
            sentence = list(self.c(sentence))
            other_sentence = list(self.c(other_sentence))
        sims = np.array(
            [cosine_similarity(self.w2v(w), self.w2v(ow))
             for w in sentence
             for ow in other_sentence]
        )
        result = sims[sims.argsort()[-self.m:]].mean()
        return 0.0 if np.isnan(result) else float(result)


class MaxContainsScorer(Scorer):
    def __init__(self, remove_punctuation=True, remove_stopwords=True,
                 do_stem=True):
        super().__init__(remove_punctuation, remove_stopwords, do_stem)

    def __call__(self, sentence: List[str], other_sentence: List[str], *args,
                 **kwargs):
        if self.c.functions:
            sentence = list(self.c(sentence))
            other_sentence = list(self.c(other_sentence))
        return sum(1 for w in sentence if w in other_sentence)


ANY = 'all'
FULL_ONLY = 'full'


def yield_ngrams(sentence, n=1, skip_punctuation=True,
                 skip_stopwords=FULL_ONLY):
    i = 0
    if skip_punctuation:
        sentence = [s for s in sentence if s not in string.punctuation]
    if skip_stopwords == ANY:
        sentence = [s for s in sentence if s not in stopwords.words('english')]
    while i + n <= len(sentence):
        ngram = sentence[i:i + n]
        if all(w in stopwords.words('english') for w in ngram):
            pass
        else:
            yield ngram
        i += 1


class ContainsUniqueNgramScorer(Scorer):
    def __init__(self, remove_punctuation=True, remove_stopwords=False,
                 do_stem=True, n=2):

        super().__init__(remove_punctuation, remove_stopwords, do_stem)
        self.n = n

    def __call__(self, sentence, other_sentence, rest_sentences, *args,
                 **kwargs):
        if self.c.functions:
            sentence = list(self.c(sentence))
            other_sentence = list(self.c(other_sentence))

        for n_gram in yield_ngrams(sentence, self.n):
            if fuzzy_subsentence(n_gram, other_sentence) and not any(
                    fuzzy_subsentence(n_gram, s) for s in rest_sentences):
                return True
        return False


def avg_tf_idf(question, supporting_facts, rest):
    tf_idf_scores = tf_idf(question, supporting_facts, rest)
    sf_scores = defaultdict(int)
    rest_scores = defaultdict(int)
    for w, scores in tf_idf_scores.items():
        for i, score in enumerate(scores[0]):
            sf_scores[i] += score
        for i, score in enumerate(scores[1]):
            rest_scores[i] += score
    num_query_terms = len(tf_idf_scores.keys())
    for i, s in sf_scores.items():
        sf_scores[i] = s / num_query_terms

    for i, s in rest_scores.items():
        rest_scores[i] = s / num_query_terms
    return sf_scores, rest_scores


def tf_idf(question, supporting_facts, rest):
    c = Curry(stem) + Curry(unpunct) + Curry(unstop)
    question = list(c(question))
    for i, sf in enumerate(supporting_facts):
        supporting_facts[i] = list(c(sf))

    for i, r in enumerate(rest):
        rest[i] = list(c(r))
    tf_idf_scores = dict()
    for w in question:
        tf_supporting_facts = []
        for sf in supporting_facts:
            # sf = list(c(sf))

            tf_supporting_facts.append(1 / len(sf) if w in sf else 0)
            logger.debug(tf_supporting_facts)
        tf_rest = []
        for r in rest:
            # r = list(c(r))

            tf_rest.append(1 / len(sf) if w in r else 0)
        n = len(supporting_facts + rest)
        d = (1 + sum(tf_supporting_facts) + sum(tf_rest))

        idf = log(n / d)
        tf_idf_scores[w] = (
            [tf * idf for tf in tf_supporting_facts],
            [tf * idf for tf in tf_rest]
        )

    return tf_idf_scores


def ratio(question, supporting_fact, rest, operation: Scorer, aggregate=max):
    sf_score = operation(question, supporting_fact, rest)
    results = []
    for i, r in enumerate(rest):
        new_rest = rest[:] + supporting_fact
        new_rest.remove(r)
        result = operation(question, r, new_rest)
        results.append(result)
    rest_score = aggregate(results)
    epsilon = 0.5
    return (epsilon + sf_score) / (rest_score + epsilon)


def filter(question, supporting_facts, rest, operation: Scorer, threshold):
    everything = supporting_facts + rest
    result = []
    for s in everything:
        new_rest = rest[:]
        new_rest.remove(s)
        score = operation(question, s, rest)
        if score > threshold:
            result.append((score, s))
    return result


def average_precision(solution: List[List[str]], gold: List[List[str]]):
    subset_result = solution[:len(gold)]
    logger.debug(subset_result)
    # return sum(1 for s in subset_result if s in gold) / len(gold)
    return sum(sum(1 for s in subset_result[:i] if s in gold) / i for i, _ in
               enumerate(subset_result, 1)) / len(gold)


def last_correct_rank(solution: List, gold: List):
    for i, candidate in enumerate(solution, 1):
        if candidate in gold:
            gold.remove(candidate)
            if not gold:
                return i
    return len(solution)


def rank(question, supporting_facts, rest, operation: Scorer,
         eval_function) -> float:
    candidates = rest + supporting_facts

    def key(candidate):
        logger.debug(candidate)
        rest = candidates[:]
        rest.remove(candidate)
        return operation(question, candidate, rest)

    result = sorted(candidates, key=key, reverse=True)
    logger.debug(result)
    return eval_function(result, supporting_facts)


def fuzzy_map_supporting_fact(supporting_fact: str,
                              sentences: List[str]) -> int:
    for i, sentence in enumerate(sentences):
        if remove_punctuation(supporting_fact).strip() == remove_punctuation(
                sentence).strip():
            return i
    raise ValueError(
        "No supporting fact mapped! {}\n{}".format(supporting_fact,
                                                   '\n'.join(sentences)))


def get_supporting_fact(annotations: List[Entity]) -> List[str]:
    return [e.surface_form for e in annotations if e.type == "SupportingFact"]


def prepare(sample: Sample, annotations: List[Entity], paragraph_extract,
            just_supp_fact_ids=False):
    supp_facts_annotations = get_supporting_fact(annotations)
    sentences = paragraph_extract(sample.raw_text)
    question = get_question(sample.raw_text)
    try:
        supp_fact_ids = [fuzzy_map_supporting_fact(ann, sentences) for ann in
                         supp_facts_annotations]
    except ValueError as e:
        if len(supp_facts_annotations) == 1:
            sf = supp_facts_annotations[0]
            try:
                fuzzy_map_supporting_fact(sf, [question])
                supp_fact_ids = []
            except ValueError:
                raise e
        else:
            raise e

    if not just_supp_fact_ids:
        supp_fact_ids = sorted(supp_fact_ids, reverse=True)

        supp_facts = [sentences.pop(i) for i in supp_fact_ids]

        return (word_tokenize(question),
                [word_tokenize(s) for s in supp_facts],
                [word_tokenize(s) for s in sentences])
    else:
        return (word_tokenize(question),
                supp_fact_ids,
                [word_tokenize(s) for s in sentences])


def run_ratio_on_sample(samples: List[Tuple[Sample, List[Entity]]],
                        paragraph_extract, operation, aggregate=max):
    results = []
    for i, s in enumerate(tqdm(samples)):
        try:
            q, sf, sents = prepare(*s, paragraph_extract)
        except ValueError as e:
            raise ValueError(f"{i}: {str(e)}")
        if not sf:
            continue
        micro_results = [ratio(q, ssf, sents, operation, aggregate) for ssf in
                         sf]
        # if not micro_results:
        results.append(sum(micro_results) / len(micro_results))
    return results


def run_rank_on_sample(samples: List[Tuple[Sample, List[Entity]]],
                       paragraph_extract, operation,
                       eval_function):
    results = []
    for i, s in enumerate(samples):
        try:
            q, sf, sents = prepare(*s, paragraph_extract)
        except ValueError as e:
            raise ValueError(f"{i}: {str(e)}")
        if sf:
            result = rank(q, sf, sents, operation, eval_function)
            results.append(result)
    return results
