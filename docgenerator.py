import sys
import inspect
import pygrank.algorithms
import pygrank.algorithms.postprocess
import pygrank.measures
import pygrank as pg

"""
THIS FILE IS AUTOMATICALLY RUN BY THE pre-commit HOOK FOR git.
Run it only if you have not created the hook in your local machine.
"""


def format(doc):
    if doc is None:
        return ""
    ret = ""
    prefix = ""
    open_example = False
    for line in doc.split("\n"):
        line = line.replace("\t", "   ").strip()
        if len(line) == 0:
            continue
        if open_example and ">>>" not in line:
            ret += "\n```\n"
            open_example = False
        line_break = "\n"
        if line.startswith("Example") and ":" in line:
            line = line+"\n\n```python"
            prefix = ""
            line_break = "\n\n"
            open_example = True
        elif len(ret) == 0:
            line_break == ""
        elif len(prefix) > 0 and ":" in line:
            line_break = "\n"+prefix
            line = "*"+line.replace(":", ":*")
        elif len(prefix) > 0 and ":" not in line:
            line_break = ""
        if line == "Attributes:" or line == "Args:" or line == "Returns:":
            prefix = " * "
            line_break = "\n\n"
        ret += line_break+line.replace(">>>", "").strip()+" "
    if open_example:
        ret += "\n```\n"
    return ret


G = next(pg.load_datasets_graph(["graph5"]))


def is_abstract(cls, *args):
    try:
        cls(*args).rank(G)
        return False
    except:
        print(cls, "not fully implemented (is either abstract class, or could not call __init__(*args).rank(G)")
        return True


def extract_attributes(text):
    ret = ""
    in_attributes = False
    for line in text.split("\n"):
        if not line.strip().startswith("* "):
            in_attributes = False
        if in_attributes:
            ret += line+"\n"
        if line.strip() == "Attributes:" or line.strip() == "Args:" or line.strip().endswith("args:"):
            in_attributes = True
    return ret


def combine_attributes(text, descriptions):
    ret = ""
    in_attributes = False
    for line in text.split("\n"):
        if not line.startswith(" * "):
            if in_attributes:
                for desc in descriptions:
                    to_add = extract_attributes(desc)
                    if to_add not in ret:  # handles case of inherited constructors
                        ret += to_add
            in_attributes = False
        ret += line+"\n"
        if line == "Attributes: " or line == "Args: ":
            in_attributes = True
    if in_attributes:
        for desc in descriptions:
            to_add = extract_attributes(desc)
            if to_add not in ret:  # handles case of inherited constructors
                ret += to_add

    return ret


def base_description(obj, abstract):
    extends = "<kbd>"+[cls.__name__ for cls in inspect.getmro(obj)][1]+"</kbd>"
    class_text = "\n### " + extends+" "+obj.__name__ + (
        "\n *Abstract class*\n\n" if abstract else "") + "\n" + format(obj.__doc__)[:-1]
    for name, method in inspect.getmembers(obj):
        if name == "__init__":
            desc = format(method.__doc__).strip()
            if len(desc) != 0:
                class_text += " The constructor " + desc[0:1].lower()+desc[1:]
    return class_text


def generate_filter_docs():
    text = "# :scroll: List of Graph Filters"
    text += "\n*This file is automatically generated with `docgenerator.py`.*\n\n" \
            "The following filters can be imported from the package `pygrank.algorithms`.\n" \
            "Constructor details are provided, including arguments inherited from and passed to parent classes.\n" \
            "All of them can be used through the code patterns presented at the library's " \
            "[documentation](documentation.md#graph-filters)." \
            " \n"

    base_descriptions = dict()
    abstract = dict()

    base_descriptions[pygrank.algorithms.abstract_filters.GraphFilter] = base_description(
        pygrank.algorithms.abstract_filters.GraphFilter, True)
    abstract[pygrank.algorithms.abstract_filters.GraphFilter] = True
    for name, obj in inspect.getmembers(sys.modules["pygrank.algorithms"]):
        if inspect.isclass(obj) and issubclass(obj, pygrank.algorithms.abstract_filters.GraphFilter):
            abstract[obj] = is_abstract(obj)
            base_descriptions[obj] = base_description(obj, abstract[obj])

    count = 0
    for abstr in base_descriptions:
        if abstract[abstr]:
            #text += "\n**"+abstr.__name__+"**\n"
            for obj in base_descriptions:
                if not abstract[obj] and abstr == list(inspect.getmro(obj))[1]:
                    count += 1
                    text += str(count)+". ["+obj.__name__+"](#"+""+[cls.__name__ for cls in inspect.getmro(obj)][1].lower()+"-"+obj.__name__.lower()+")\n"

    preprocessor_descriptions = [format(pygrank.preprocessor.__doc__), format(pygrank.algorithms.filters.ConvergenceManager.__init__.__doc__)]

    for abstr in base_descriptions:
        if abstract[abstr]:
            for obj in base_descriptions:
                if not abstract[obj] and abstr in inspect.getmro(obj):
                    text += combine_attributes(base_descriptions[obj], ([base_descriptions.get(cls, "") for cls in inspect.getmro(obj)][1:])+preprocessor_descriptions)

    with open("documentation/graph_filters.md", "w") as file:
        file.write(text)


def generate_postprocessor_docs():
    text = "# :scroll: List of Postprocessors"
    text += "\n*This file is automatically generated with `docgenerator.py`.*\n\nThe following postprocessors can be imported from the package " \
            "`pygrank.algorithms.postprocess`.\n" \
            "Constructor details are provided, including arguments inherited from and passed to parent classes.\n" \
            "All of them can be used through the code patterns presented at the library's [documentation](documentation.md#postprocessors). " \
            " \n"

    base_descriptions = dict()
    abstract = dict()

    for name, obj in inspect.getmembers(sys.modules["pygrank.algorithms.postprocess"]):
        if inspect.isclass(obj) and issubclass(obj, pygrank.algorithms.postprocess.Postprocessor):
            abstract[obj] = False#is_abstract(obj, pygrank.algorithms.PageRank()) TODO - fix this
            base_descriptions[obj] = base_description(obj, abstract[obj])
    base_descriptions[pygrank.algorithms.postprocess.Postprocessor] = base_description(
        pygrank.algorithms.postprocess.Postprocessor, True)
    abstract[pygrank.algorithms.postprocess.Postprocessor] = True

    count = 0
    for obj in base_descriptions:
        if not abstract[obj]:
            count += 1
            text += str(count)+". ["+obj.__name__+"](#"+""+[cls.__name__ for cls in inspect.getmro(obj)][1].lower()+"-"+obj.__name__.lower()+")\n"

    for abstr in base_descriptions:
        if abstract[abstr]:
            for obj in base_descriptions:
                if not abstract[obj] and abstr in inspect.getmro(obj):
                    text += combine_attributes(base_descriptions[obj], [base_descriptions.get(cls,"") for cls in inspect.getmro(obj)][1:])

    with open("documentation/postprocessors.md", "w") as file:
        file.write(text)


def generate_metric_docs():
    text = "# :scroll: List of Measures"
    text += "\n*This file is automatically generated with `docgenerator.py`.*\n\nThe following measures can be imported from the package " \
            "`pygrank.measures`.\n" \
            "Constructor details are provided, including arguments inherited from and passed to parent classes.\n" \
            "All of them can be used through the code patterns presented at the library's [documentation](documentation.md#evaluation). " \
            " \n"

    base_descriptions = dict()
    abstract = dict()

    for name, obj in inspect.getmembers(sys.modules["pygrank.measures"]):
        if inspect.isclass(obj) and issubclass(obj, pygrank.measures.Measure):
            abstract[obj] = False  #is_abstract(obj, pygrank.algorithms.PageRank()) TODO - fix this
            base_descriptions[obj] = base_description(obj, abstract[obj])

    base_descriptions[pygrank.measures.Supervised] = base_description(pygrank.measures.Supervised, True)
    abstract[pygrank.measures.Supervised] = True
    base_descriptions[pygrank.measures.Unsupervised] = base_description(pygrank.measures.Unsupervised, True)
    abstract[pygrank.measures.Unsupervised] = True
    base_descriptions[pygrank.measures.Measure] = base_description(pygrank.measures.Measure, True)
    abstract[pygrank.measures.Measure] = True
    base_descriptions[pygrank.measures.MeasureCombination] = base_description(pygrank.measures.MeasureCombination, True)
    abstract[pygrank.measures.MeasureCombination] = True

    count = 0
    for abstr in base_descriptions:
        if abstract[abstr]:
            for obj in base_descriptions:
                if not abstract[obj] and abstr == list(inspect.getmro(obj))[1]:
                    count += 1
                    text += str(count)+". ["+obj.__name__+"](#"+""+[cls.__name__ for cls in inspect.getmro(obj)][1].lower()+"-"+obj.__name__.lower()+")\n"

    for abstr in base_descriptions:
        if abstract[abstr]:
            for obj in base_descriptions:
                if not abstract[obj] and abstr == list(inspect.getmro(obj))[1]:
                    text += combine_attributes(base_descriptions[obj], [base_descriptions.get(cls,"") for cls in inspect.getmro(obj)][1:])

    with open("documentation/measures.md", "w") as file:
        file.write(text)


def generate_tuning_docs():
    text = "# :scroll: List of Tuners"
    text += "\n*This file is automatically generated with `docgenerator.py`.*\n\nThe following tuning mechanisms can be imported from the package " \
            "`pygrank.algorithms.autotune`.\n" \
            "Constructor details are provided, including arguments inherited from and passed to parent classes.\n" \
            "All of them can be used through the code patterns presented at the library's [documentation](documentation.md#autotune). " \
            " \n"

    base_descriptions = dict()
    abstract = dict()

    for name, obj in inspect.getmembers(sys.modules["pygrank.algorithms.autotune"]):
        if inspect.isclass(obj) and issubclass(obj, pygrank.algorithms.autotune.Tuner):
            abstract[obj] = False#is_abstract(obj, pygrank.algorithms.PageRank()) TODO - fix this
            base_descriptions[obj] = base_description(obj, abstract[obj])

    base_descriptions[pygrank.algorithms.autotune.Tuner] = base_description(pygrank.algorithms.autotune.Tuner, True)
    abstract[pygrank.algorithms.autotune.Tuner] = True

    count = 0
    for abstr in base_descriptions:
        if abstract[abstr]:
            for obj in base_descriptions:
                if not abstract[obj] and abstr == list(inspect.getmro(obj))[1]:
                    count += 1
                    text += str(count)+". ["+obj.__name__+"](#"+""+[cls.__name__ for cls in inspect.getmro(obj)][1].lower()+"-"+obj.__name__.lower()+")\n"

    for abstr in base_descriptions:
        if abstract[abstr]:
            for obj in base_descriptions:
                if not abstract[obj] and abstr == list(inspect.getmro(obj))[1]:
                    text += combine_attributes(base_descriptions[obj], [base_descriptions.get(cls,"") for cls in inspect.getmro(obj)][1:])

    with open("documentation/tuners.md", "w") as file:
        file.write(text)


def generate_dataset_docs():
    text = "# :scroll: List of Datasets"
    text += "\n*This file is automatically generated with `docgenerator.py`.*\n\nThe following datasets are automatically " \
            "downloaded if not found when trying to load them. Please visit respective sources to learn how to cite them " \
            "in new research.\n" \
            "All datasets can be imported with code patterns presented at the library's [documentation](documentation.md#datasets). " \
            " \n"
    text += "\n| Dataset | Source | Graph | Node labels | Node features |\n"
    text += "| --- | --- | --- | --- | --- |\n"
    for name, source in pg.datasets.items():
        text += "| "+name
        text += " | "+source.get("url", "**UNKNOWN SOURCE**")
        text += " | "
        if "pairs" in source or "all" in source:
            text += ":heavy_check_mark:"
        text += " | "
        if "labels" in source or "all" in source or "groups" in source or "features" in source:
            text += ":heavy_check_mark:"
        text += " | "
        if "features" in source or "all" in source:
            text += ":heavy_check_mark:"
        text += " |\n"

    with open("documentation/datasets.md", "w") as file:
        file.write(text)


if __name__ == '__main__':
    generate_filter_docs()
    generate_postprocessor_docs()
    generate_metric_docs()
    generate_tuning_docs()
    generate_dataset_docs()