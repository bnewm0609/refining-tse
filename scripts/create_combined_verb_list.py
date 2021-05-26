# Downloads and scrapes the Giant-VerbList PDF for experiments
import pdftotext
import re
import shlex
import shutil
import subprocess

DOWNLOAD_CMD = "wget -O data/verbs/Giant-Verb-List-3250-Verbs.pdf https://patternbasedwriting.com/1/Giant-Verb-List-3250-Verbs.pdf"


def conjugate(verb):
    verb = verb.strip().lower()
    if verb == "have":
        verb_z = "has"
    elif verb.endswith('y') and verb[-2] not in 'aeiou':
        verb_z = verb[:-1] + "ies"
    elif verb.endswith(("sh", "s", "x", "ch", "z", "o")):
        verb_z = verb + "es"
    else:
        verb_z = verb + "s"
    return verb_z


def load_pdf():
    with open("data/verbs/Giant-Verb-List-3250-Verbs.pdf", "rb") as f:
        pdf = pdftotext.PDF(f)

    # pages 6-31 have the verbs we need
    verb_lines = []
    for i in range(6, 32):
        for line in pdf[i].split("\n"):
            if re.match(r"\d+\.", line.strip()) is not None:
                verb_lines.extend(line.strip().split())

    # filter out the verbs from the numbers and other formatting
    # Format of verb_lines is roughly:
    #  >>> ['1.', 'VERB_1', '2.', 'VERB_2', '*', '3.', 'VERB_3', ...]
    i = 0
    verbs = []
    while i < len(verb_lines):
        if re.match(r"\d+\.", verb_lines[i]) is not None:
            # the next element is a verb
            verbs.append(verb_lines[i + 1])
            i += 1
        i += 1

    assert len(verbs) == 3250
    return verbs


def main():
    # download the pdf
    print("Downloading The Giant Verb List pdf...")
    subprocess.run(shlex.split(DOWNLOAD_CMD))
    # load the pdf
    verbs = load_pdf()

    # write the verbs from the pdf to disk
    out_lines = []
    for verb in verbs:
        if verb == "be":  # special case - skip because it's in the old combined_verb_list.csv
            continue
        if verb == "breath":  # special case
            verb = "breathe"
        out_lines.append(",".join(["_", conjugate(verb), verb, "_"]))

    # now combine them with the rest of the verbs:
    with open("data/verbs/combined_verb_list.csv") as f:
        old_verbs = [line.strip() for line in f.readlines()][1:]

    shutil.copyfile("data/verbs/combined_verb_list.csv", "data/verbs/.old_combined_verb_list.csv")  # save the old list
    out_lines = sorted(set(out_lines + old_verbs))  # combine the two lists
    out_lines = [",".join(["idx", "sing", "plur", "freq"])] + out_lines
    with open("data/verbs/combined_verb_list.csv", "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
