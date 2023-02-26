import spacy
from spacy.tokens import SpanGroup


def process_text(text):
    """
    Processes text with two different models and returns the results.
    Args:
        text (str): Text to process.
    Returns:
        A tuple containing:
            doc1 (spacy.tokens.doc.Doc): Doc object from the default pipeline.
            doc2 (spacy.tokens.doc.Doc): Doc object from the coref pipeline.
    """

    nlp1 = spacy.load("en_core_web_sm")
    doc1 = nlp1(text)

    nlp2 = spacy.load("en_coreference_web_trf")
    doc2 = nlp2(text)

    return (doc1, doc2)


def read_file(file_path):
    """
    Reads a file and returns a list of strings, where each string is a line in the file.
    Args:
        file_path (str): Path to the file.
    Returns:
        (list): List of strings.
    """

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f]

    return lines


# Adapted from: https://gist.github.com/thomashacker/b5dd6042c092e0a22c2b9243a64a2466
def resolve_references(text: str, print_clusters: bool = False) -> str:
    """
    Resolves coreferences
    Args:
        text (str): Text to process
        print_clusters (bool): If true, will print the clusters
    Returns:
        (str): Text with resolved references
    """

    doc1, doc2 = process_text(text)

    if print_clusters:
        print("Clusters:")
        for cluster in doc2.spans:
            print(f"{cluster}: {doc2.spans[cluster]}")

    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc2.spans.items() if key.startswith("coref_cluster")
    ]

    # If a in a cluster span is a verb, remove the cluster
    for cluster in clusters:
        for span in cluster:
            if doc1[span.start].pos_ == "VERB":
                # print("Verb cluster:", cluster)
                clusters.remove(cluster)
                break

    ignore_words = read_file("src/coreference_resolution/ignore_words.txt")

    new_clusters = []
    for i, cluster in enumerate(clusters):
        new_cluster = SpanGroup(doc2, name=f"coref_cluster_{i+1}")
        for span in cluster:
            if span.text.lower() not in ignore_words:
                new_cluster.append(span)
        new_clusters.append(new_cluster)

    # Iterate through every found cluster
    for cluster in new_clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = (
                first_mention.text + mention_span[0].whitespace_
            )

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc2:
        # Check if token exists in token_mention_mapper; ignore pronouns
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


with open("src/coreference_resolution/sample_texts/text_0.txt", "r") as file:
    text = file.read()

print(resolve_references(text))