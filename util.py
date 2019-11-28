import re
from collections import defaultdict
from typing import List

import glob, os

try:
    name = get_ipython().__class__.__name__
    if name == "ZMQInteractiveShell":
        from tqdm import tqdm_notebook as tqdm

    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


def read_schema(path='annotation.conf'):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    start = lines.index('[entities]') + 1
    end = lines.index('[events]')
    lines = lines[start:end]

    labels = []
    categories = defaultdict(list)
    indent = 0
    active_categories = list()
    for line in lines:
        if line.strip() and not line.startswith("#"):

            new_indent = sum(1 for a in line if a == '\t')
            if new_indent < indent:
                for _ in range(indent - new_indent):
                    active_categories.pop()

            indent = new_indent

            line = line.strip()

            for category in active_categories:
                categories[category[1:]].append(
                    line[1:] if line.startswith("!") else line)

            if line.startswith('!'):
                active_categories.append(line)
            else:
                labels.append(line)
    return labels, categories


class Sample:
    def __init__(self, idx, raw_text):
        self.raw_text = raw_text
        self.index = idx


class Entity:
    def __init__(self, id,
                 type, start_index, end_index, surface_form):
        self.id = id
        self.type = type
        self.start_index = start_index
        self.end_index = end_index
        self.surface_form = surface_form

    def __repr__(self):
        return f"{self.id} {self.type} [{self.start_index}:{self.end_index}] '{self.surface_form}'"


def read_collection_annotation(path, entities_only=True):
    collection = []
    print(os.path.join(path, "*.txt"))
    for f in sorted(glob.glob(os.path.join(path, "*.txt"))):
        with open(f, 'r') as file:
            lines = file.read()
        sample = Sample(idx=int(f.split('.', 1)[0].rsplit('/', 1)[-1]),
                        raw_text=lines)
        with open(f[:-4] + ".ann") as file:
            lines = file.read().splitlines()
        annotations = []
        lines = (l for l in lines)
        for line in lines:
            line = line.replace('\t', ' ')
            if line.startswith("T"):  # Entity
                annotations.append(Entity(*line.split(" ", 4)))
            elif line.startswith("R") and not entities_only:
                # TODO
                ...
        collection.append((sample, annotations))
    return collection
    # for


from collections import Counter


def count_category(collection, category, categories, binary=True,
                   per_samples=True,
                   combine_categories=None,
                   only_if_sf_present=False,
                   only_if_not_unanswerable=False):
    counter = Counter()
    combine_categories = combine_categories or []
    for (sample, annotations) in collection:
        local_counter = Counter()
        for annotation in annotations:
            if annotation.type in categories[category]:
                combined = False
                for to_combine in combine_categories:
                    if annotation.type in categories[to_combine]:
                        local_counter[to_combine] = 1 if binary else \
                            local_counter[category] + 1
                        combined = True
                if not combined:
                    local_counter[annotation.type] = 1 if binary else \
                        local_counter[category] + 1
        counter += local_counter
    if per_samples:
        if only_if_sf_present:
            total = sum(any(a.type == "SupportingFact" for a in aa) for _, aa in
                        collection)
        elif only_if_not_unanswerable:
            total = sum(all(a.type != "Unanswerable" for a in aa) for _, aa in
                        collection)
        else:
            total = len(collection)
    else:
        total = sum(v for k, v in counter.items())
    return counter, Counter({k: v / total for k, v in counter.items()})


def count_reasoning(collection, categories):
    return count_category(collection, "Reasoning",
                          combine_categories=['Operations'],
                          categories=categories)


def count_answers(collection, categories):
    return count_category(collection, "Answer", categories=categories)


def count_linguistic_features(collection, categories):
    return count_category(collection, "LinguisticComplexity",
                          categories=categories)


def format_counter_for_latex(counter, categories):
    return " ".join(
        f"({i + 0.5},{counter.get(category, 0):.0%}"[:-1] + ')' for i, category
        in enumerate(categories))


no_name = """
    xticklabels={%s},
    xtick={%s},
    xmin=0, xmax=%d,
    ]
    \\addplot+[bar shift=-6.25pt, color=bblue, fill=bblue] 
coordinates {
    %s
    };
\\addplot+[bar shift=-3.75pt, color=rred, fill=rred] 
coordinates {
    %s
    };
\\addplot+[bar shift=-1.25pt, color=ggreen, fill=ggreen] 
coordinates {
    %s
    };
\\addplot+[bar shift=1.25pt, color=ppurple, fill=ppurple] 
coordinates {
    %s
    };
\\addplot+[bar shift=3.75pt, color=pink, fill=pink] 
coordinates {
    %s
    };
\\addplot+[bar shift=6.25pt, color=teal, fill=teal] 
coordinates {
    %s
    };
    """

name = """
    xticklabels={%s},
    xtick={%s},
    xmin=0, xmax=%d,
        ]
    \\addplot+[legend entry=\\textsc{MSMarco}, color=bblue, fill=bblue] 
coordinates {
    %s
    };
\\addplot+[legend entry=\\textsc{HotpotQA}, color=rred, fill=rred] 
coordinates {
    %s
    };
\\addplot+[legend entry=\\textsc{ReCoRd}, color=ggreen, fill=ggreen] 
coordinates {
    %s
    };
\\addplot+[legend entry=\\textsc{MultiRC}, color=ppurple, fill=ppurple] 
coordinates {
    %s
    };
\\addplot+[legend entry=\\textsc{NewsQA}, color=pink, fill=pink] 
coordinates {
    %s
    };
\\addplot+[legend entry=\\textsc{DROP}, color=teal, fill=teal] 
coordinates {
    %s
    };
    """


def format_counters_for_table(categories, counters):
    result = []
    for category in categories:
        result.append(f"{category} & " + " & ".join(
            f"{c1[category]} & {c2[category] * 100:.1f}" for c1, c2 in
            counters) + "\\\\")
    return "\n".join(result)


def tp_fp_fn_per_sample(retrieved: List[Entity], gold: List[Entity],
                        categories):
    gold = {x.type for x in gold if x.type in categories}
    retrieved = {x.type for x in retrieved if x.type in categories}
    if not gold and not retrieved:
        return 1, 1, 0
    return sum(1 for r in retrieved if r in gold), sum(
        1 for r in retrieved if r not in gold), sum(
        1 for r in gold if r not in retrieved)


def interpolated_agreement_tp_fp_fn(first_annotation, second_annotation,
                                    categories=None):
    # assuming: first annotation is the full one
    annotation_pairs = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for s, a in second_annotation:
        annotation_pairs.append((first_annotation[s.index], (s, a)))
    for ((s, as1), (_, as2)) in annotation_pairs:
        tp, fp, fn = tp_fp_fn_per_sample(as2, as1, categories)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return total_tp, total_fp, total_fn


def format_all_for_latex(cats, a, b, c, d, e, f, map_to=None, use_names=False):
    lookup_map = {}
    if map_to:
        lookup_map.update(map_to)
    cats_str = ",".join(lookup_map.get(c, c) for c in cats)
    ticks_str = ",".join(str(i) for i in range(len(cats) + 1))
    xmax = len(cats)
    if use_names:
        return name % (cats_str, ticks_str, xmax, a, b, c, d, e, f)
    else:
        return no_name % (cats_str, ticks_str, xmax, a, b, c, d, e, f)


def split_hotpotqa(text: str) -> List[str]:
    text = text.split("Paragraph:\n", 1)[1].split("\nQuestion:\n", -1)[
        0].strip()
    return [l.split(":", 1)[1].strip() for l in text.splitlines() if ":" in l]


def split_drop(text: str) -> List[str]:
    return text.split("Paragraph:", 1)[-1].split('Question:', 1)[
        0].strip().splitlines()


def split_newsqa(text: str) -> List[str]:
    r = [t for t in text.split("Paragraph:", 1)[-1].split('Question:', 1)[
        0].strip().splitlines() if t]
    if re.match('(.*) --', r[0]):
        r[0] = r[0].split("--", 1)[1].strip()
    return r


def split_msmarco(text: str) -> List[str]:
    r = [t for t in text.split("Paragraph:", 1)[-1]
        .split('Question:', 1)[0].strip().splitlines() if t]
    if r[0].startswith("(!)"):
        r[0] = r[0].split("(!)", 1)[1].strip()
    return r


def split_multirc(text: str) -> List[str]:
    text = text.split("Paragraph:\n", 1)[1].split("\nQuestion:\n", -1)[
        0].strip()
    return [l.split(":", 1)[1].strip() for l in text.splitlines() if ":" in l]


def split_record(text: str) -> List[str]:
    r = [t for t in text.split("Paragraph:", 1)[-1]
        .split('Question:', 1)[0].strip().splitlines() if t]
    if '(CNN)' in r[0]:
        r[0] = r[0].split("(CNN)", 1)[1].strip()
    return r


def get_question(text: str) -> str:
    return text.split("Question:\n", 1)[1].split("\n", 1)[0]
