"""Defines plotting code for aggregating experiment results"""
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.ticker import AutoLocator
import numpy as np
import os

matplotlib.use("agg")
plt.rcParams["font.family"] = "serif"

RESULTS_DIR = "results"
PARSE_LINES_FROM_END = 20

# MPE templates to display in main paper results figure
few_templates = [
    "ML_simple_agrmt",
    "ML_obj",
    "ML_obj_rel_no_comp_across",
    "blimp",
]

# MPE template names in the order they should appear in the main paper results table
ordered_templates = [
    "ML_simple_agrmt",
    "ML_sent_comp",
    "ML_vp_coord",
    "ML_prep",
    "ML_subj",
    "ML_obj",
    "ML_obj_rel_no_comp_across",
    "ML_obj_rel_within",
    "ML_obj_rel_no_comp_within",
    "blimp",
]

# Map huggingface names to results table column headers
model_map = {
    "bert-base-cased": "BERT cased",
    "bert-base-uncased": "BERT uncased",
    "roberta-base": "RoBERTa",
    "gpt2": "GPT2",
    "bert-large-cased": "BERT cased (large)",
    "bert-large-uncased": "BERT uncased (large)",
    "roberta-large": "RoBERTa (large)",
    "gpt2-xl": "GPT2 (XL)",
}


# Map MPE template names to result table display names
template_map = {
    "ML_simple_agrmt": "Simple",
    "ML_sent_comp": "In a sentential complement",
    "ML_vp_coord": "VP coordination",
    "ML_obj_rel_no_comp_across": "Across object relative (no that)",
    "ML_obj_rel_within": "In object relative clause",
    "ML_obj_rel_no_comp_within": "In object relative (no that)",
    "ML_prep": "Across prepositional phrase",
    "ML_obj": "Across object relative clause",
    "ML_subj": "Across subject relative clause",
    "blimp": "BLiMP",
}

# Map internal metric names to result table display names
metric_name_map = {
    "mw.": "MW",
    "tse_ew.": "EW",
    "tse_ew.ml": "TSE",
}

# Map internal statistic names to appendix plot axis names
FIELD_MAP = {
    "probcounted": "Probability Counted",
    "cutoffs": "Cutoff",
    "scores": "Score",
    "invalid": "% Invalid",
}

def parse_csv(line):
    line = line.split(":")[1].strip()
    return [float(n) for n in line.split(",")]

def parse_line(line, metric):
    """Parses statistics from a line an experiment log file"""
    if "top-k" in line:
        return f"top-k.{metric}", parse_csv(line)
    elif "bottom-k" in line:
        return f"bottom-k.{metric}", parse_csv(line)
    else:
        return f"ml.{metric}", parse_csv(line)

def get_npz(line, in_metric, out_metric, f):
    """Extracts statistics from npz files"""
    if "top-k" in line:
        in_metric = f"{in_metric} (top-k)"
        out_metric = f"top-k.{out_metric}"
    elif "bottom-k" in line:
        in_metric = f"{in_metric} (bottom-k)"
        out_metric = f"bottom-k.{out_metric}"
    path = os.path.join(*f.split(os.path.sep)[:-2], "npzs", f"{in_metric}.npz")
    arr = np.load(path)['arr_0']
    return out_metric, arr

def parse_main_file(f):
    """Parses the metrics out of a 'main' experiment log file (for MW/EW)"""
    results_dict = {}
    with open(f) as mainfile:
        # All aggregate statistics are at the end of the file
        lines = mainfile.readlines()[-PARSE_LINES_FROM_END:]
        for line in lines:
            if line.startswith("Cutoffs"):
                key, vals = parse_line(line, "cutoffs")
            elif line.startswith("Overall scores"):
                key, vals = parse_line(line, "scores")
            elif "Prob counted" in line:
                key, vals = get_npz(line,  "Prob counted", "probcounted", f)
                vals = np.nanmean(vals, axis=0).tolist()
            elif line.startswith(r"% examples invalid"):
                key, vals = parse_line(line, "invalid")
            else:
                continue
            results_dict[key] = vals
    return results_dict
            

def parse_ml_file(f):
    """Parses the metrics out of an 'ml' experiment log file (for TSE)"""
    results_dict = {}
    with open(f) as mlfile:
        lines = mlfile.readlines()[-PARSE_LINES_FROM_END:]
        for line in lines:
            if line.startswith("Overall model score"):
                key, vals = parse_line(line, "scores")
            elif line.startswith("Number of examples"):
                key, vals = parse_line(line, "numexamples")
            else:
                continue
            results_dict[key] = vals
    return results_dict

def get_experiment_label(path):
    """Extracts short experiment name from file path"""
    label = path.split("/")
    label = ".".join([*label[2:4], label[5]])
    label = label.replace(".metrics", "")
    label = label.replace(".main", "")
    return label

def get_experiment_label_long(path):
    """Extracts long experiment name from file path"""
    label = path.split("/")
    label = ".".join([*label[1:4], label[5]])
    label = label.replace(".metrics", "")
    label = label.replace(".main", "")
    return label


def generate_results_table(results_dict_all, blacklist=None):
    """Generates main results table latex."""
    scores_dict = {}
    metric_names = []
    models = []
    for experiment, results_dict in results_dict_all.items():
        val = results_dict.get("top-k.scores")
        val = results_dict.get("ml.scores") if val is None else val
        if val is None:
            continue
        
        x = get_experiment_label_long(experiment).split(".")
        if len(x) == 3:
            model, templates, metric_1 = x
            metric_2 = ""
        elif len(x) == 4:
            model, templates, metric_1, metric_2 = x
        
        metric_name = ".".join([metric_1, metric_2])
        
        if blacklist is not None and metric_name in blacklist:
            continue
        
        if metric_name not in metric_names:
            metric_names.append(metric_name)
        d = scores_dict.get((templates, model), {})
        d.update({metric_name: val[-1]})
        scores_dict[(templates, model)] = d
        if model not in models:
            models.append(model)

    # reorder metric_names
    reordered_metric_names = []
    if "mw." in metric_names:
        reordered_metric_names.append("mw.")
    if "tse_ew." in metric_names:
        reordered_metric_names.append("tse_ew.")
    if "tse_ew.ml" in metric_names:
        reordered_metric_names.append("tse_ew.ml")
    metric_names = reordered_metric_names

        
    begin_table_str = r"\begin{tabular}{c "
    for model in models:
        begin_table_str += "| ccc "
    begin_table_str += "}"
    result = [begin_table_str, r"\toprule"]
    # Generate the table title
    title_str = "Templates "
    for model in models:
        c_fmt = "c" if model == models[0] else "|c"
        title_str += f"& \\multicolumn{{3}}{{{c_fmt}}}{{{model_map[model]}}}"
    title_str += r"\\"
    result.append(title_str)
    sub_title_str = ""
    for model in models:
        for metric_name in metric_names:
            sub_title_str += f"& {metric_name_map[metric_name]} "
    sub_title_str += r"\\"
    result.append(sub_title_str)
    result.append(r"\midrule")

    # generate the table rows
    for templates in ordered_templates:
        row = f"{template_map[templates]} "
        for model in models:
            if (templates, model) not in scores_dict:
                row += "& - " * 3
                continue
            scores = scores_dict[(templates, model)]
            for metric_name in metric_names:
                if metric_name not in scores:
                    row += "& - "
                    continue
                if metric_name == "mw.":
                    row += f"& \\textit{{{scores[metric_name]:.2f}}}"
                elif f"{scores[metric_name]:.2f}" == f"{max([scores[s] for s in scores if s != 'mw.']):.2f}":
                    row += f"& \\textbf{{{scores[metric_name]:.2f}}} "
                else:
                    row += f"& {scores[metric_name]:.2f} "
        row += r"\\"
        result.append(row.replace("_", r"\_"))
    result.append(r"\bottomrule")
    result.append(r"\end{tabular}")
    return result


def generate_top_bottom_plot(results_dict_all, metric, exp, fields=("cutoffs", "scores"), ml=True, title=None):
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure("111", dpi=200)
    ax_top = fig.gca()
    top_bottom_plot_helper(results_dict_all,
                             ax_top,
                             metric,
                             exp,
                             fields,
                             ml,
                             title,
                             display_bottom_ax=True,
                             display_y_ax=True,
                             display_top_ax=True,
                             plot_subset=True)
    return fig

def top_bottom_plot_helper(results_dict_all, ax_top, metric, exp, fields, ml, title, legend=True, display_bottom_ax=False, display_y_ax=False, display_top_ax=False, plot_subset=False):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax_bottom = ax_top.twiny()
    ax_bottom.set_xscale('log')
    ax_top.set_ylim(0, 1)

    templates = few_templates if plot_subset else ordered_templates

    seen_labels = set()

    legend_labels = {}
    for i, template in enumerate(templates):
        experiment = ".".join([exp, template, metric])
        results_dict = results_dict_all.get(experiment)
        if not results_dict:
            continue
        xs_top = results_dict.get(f"top-k.{fields[0]}")
        ys_top = results_dict.get(f"top-k.{fields[1]}")
        xs_bottom = results_dict.get(f"bottom-k.{fields[0]}")
        ys_bottom = results_dict.get(f"bottom-k.{fields[1]}")

        style = "solid"

        if not xs_top or not ys_top:
            if ml and results_dict.get("ml.scores"):
                xs = [1e-7, 1]
                ys = results_dict.get("ml.scores") * 2
                style = "dashed"
                marker = None
            else:
                continue

        label = experiment
        if label in seen_labels:
            continue
        else:
            seen_labels.add(label)
        t, = ax_top.plot(xs_top, ys_top, label=f"{label}.top", marker="o", linestyle="solid", color = colors[i])
        b, = ax_bottom.plot(xs_bottom, ys_bottom, label=f"{label}.bottom", marker="x", linestyle="dashed", color=colors[i])
        legend_labels[template] = ax_top.plot([], [], color=colors[i], marker="s", ls="none")


    ax_top.xaxis.set_ticks_position("top")

    ax_bottom.xaxis.set_ticks_position("bottom")
    ax_bottom.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom

    if display_bottom_ax:
        ax_bottom.set_xlabel(f"Bottom {FIELD_MAP[fields[0]]}", fontsize=12)
        ax_bottom.set_ylim(0, 1)
    else:
        ax_bottom_labels = [item.get_text() for item in ax_bottom.get_xticklabels()]
        ax_bottom.set_xticklabels(['']*len(ax_bottom_labels))

    if display_top_ax:
        ax_top.xaxis.set_label_position("top")
        ax_top.set_xlabel(f"Top {FIELD_MAP[fields[0]]}", fontsize=12)
    else:
        ax_top_labels = [item for item in ax_top.get_xticklabels()]
        for lab in ax_top_labels:
            lab.set_visible(False)

    if display_y_ax:
        ax_top.set_ylabel(FIELD_MAP[fields[1]], fontsize=12)

    # configure the legend
    template_names_sorted = [template_map[temp_name] for temp_name in templates if legend_labels.get(temp_name) is not None] + ["Top", "Bottom"]
    legend_labels_sorted = [legend_labels[temp_name][0] for temp_name in templates if legend_labels.get(temp_name) is not None]
    legend_labels_sorted += [
        ax_top.plot([], [], marker="o", color="black", ls="solid")[0],
        ax_top.plot([], [], marker="x", color="black", ls="dashed")[0],
    ]

    title = title if title else exp.captialize()
    if legend:
        leg_loc = (0.01, 0.01) if plot_subset else (1.04, 0)
        leg_title = title if plot_subset else None
        leg = ax_bottom.legend(
            legend_labels_sorted,
            template_names_sorted,
            title=title,
            loc=leg_loc)
        # place the title over the legend
        y_title = 0.45 if plot_subset else 0.85
        if plot_subset:
            plt.setp(leg.get_title(),fontsize=14)
        else:
            ax_top.set_title(title, fontsize=16, x=1.325, y=y_title)

    else:
        ax_top.set_title(title, fontsize=16)

    return ax_top, ax_bottom, (legend_labels_sorted, template_names_sorted)

def generate_sup_plots_scores(results_dict, fields=("cutoffs", "scores")):
    """Generate the plots for the supplementary materials that include MW and EW
        scores for all cutoffs and all template types.
    """
    fig, ax = plt.subplots(nrows=4, ncols=2, sharex="col", sharey="row", dpi=200, figsize=(16,16))
    params = [
        [{"metric": "tse_ew", "exp": "bert-large-uncased", "title":"BERT uncased (EW)"},
        {"metric": "mw", "exp": "bert-large-uncased", "title":"BERT uncased (MW)"}],
        [{"metric": "tse_ew", "exp": "bert-large-cased", "title":"BERT cased (EW)"},
        {"metric": "mw", "exp": "bert-large-cased", "title":"BERT cased (MW)"}],
        [{"metric": "tse_ew", "exp": "roberta-large", "title":"RoBERTa (EW)"},
        {"metric": "mw", "exp": "roberta-large", "title":"RoBERTa (MW)"}],
        [{"metric": "tse_ew", "exp": "gpt2-xl", "title":"GPT2 (EW)"},
        {"metric": "mw", "exp": "gpt2-xl", "title":"GPT2 (MW)"}],
    ]

    for x, y in np.ndindex(ax.shape):
        kwargs = {}
        if x == ax.shape[0] - 1:
            kwargs['display_bottom_ax'] = True
        elif x == 0:
            kwargs['display_top_ax'] = True
        if y == 0:
            kwargs['display_y_ax'] = True

        _, _, legend_info = top_bottom_plot_helper(results_dict, ax[x,y], fields=fields, ml=False, legend=False, **params[x][y], **kwargs)

    legend_labels_sorted, template_names_sorted = legend_info
    ax[-1, -1].legend(
            legend_labels_sorted,
            template_names_sorted,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            bbox_to_anchor=(0.2, -0.2)
    )

    return fig

def generate_sup_plots_2x2(results_dict, fields=("cutoffs", "scores"), title=None):
    """Generates plots for the Supplementary Materials that need to be in a 2x2 grid.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", dpi=200, figsize=(16,10))
    params = [
        [{"metric": "tse_ew", "exp": "bert-large-cased", "title":"BERT cased"},
         {"metric": "tse_ew", "exp": "gpt2-xl", "title":"GPT2"}],
        [{"metric": "tse_ew", "exp": "bert-large-uncased", "title":"BERT uncased"},
         {"metric": "tse_ew", "exp": "roberta-large", "title":"RoBERTa"}],
    ]
    for x, y in np.ndindex(ax.shape):
        kwargs = {}
        if x == ax.shape[0] - 1:
            kwargs['display_bottom_ax'] = True
        elif x == 0:
            kwargs['display_top_ax'] = True
        if y == 0:
            kwargs['display_y_ax'] = True

        _, _, legend_info = top_bottom_plot_helper(results_dict, ax[x,y], fields=fields, ml=False, legend=False, **params[x][y], **kwargs)

    legend_labels_sorted, template_names_sorted = legend_info
    ax[-1, -1].legend(
            legend_labels_sorted,
            template_names_sorted,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            bbox_to_anchor=(0.2, -0.2)
    )
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig

def generate_figures():
    """Main method for generating latex table and plots"""
    os.makedirs("plots", exist_ok=True)
    exps_to_path = {}
    results_dict_all = {}

    for path, directories, files in os.walk(RESULTS_DIR):
        if "metrics" in path:
            exp_name = get_experiment_label_long(path)
            if exps_to_path.get(exp_name) and exps_to_path.get(exp_name) > path:
                continue
            exps_to_path[exp_name] = path
            for file_name in files:
                if file_name == "main.txt":
                    results_dict_all[f"{path}.main"] = parse_main_file(os.path.join(path, file_name))
                elif file_name == "ML.txt":
                    results_dict_all[f"{path}.ml"] = parse_ml_file(os.path.join(path, file_name))
    results_dict_all_labeled = {
        get_experiment_label_long(experiment): scores for experiment, scores in results_dict_all.items()
    }

    latex_lines = generate_results_table(results_dict_all)
    with open("plots/latex_table.tex", "w") as f:
        f.write("\n".join(latex_lines))

    # generate plots for main paper
    fig_mw = generate_top_bottom_plot(results_dict_all_labeled, "mw", "bert-large-cased", ml=False, title="BERT (large) uncased (MW)")
    fig_mw.savefig("plots/bert-large-cased-mw.png")
    fig_mw.clf()
    fig_ew = generate_top_bottom_plot(results_dict_all_labeled, "tse_ew", "bert-large-cased", ml=False, title="BERT (large) uncased (EW)")
    fig_ew.savefig("plots/bert-large-cased-ew.png")
    fig_mw.clf()

    # generate plots for supplementary materials
    score_fig = generate_sup_plots_scores(results_dict_all_labeled)
    score_fig.savefig("plots/supp_top-bottom-scores.png", bbox_inches='tight')

    prob_counted_fig = generate_sup_plots_2x2(results_dict_all_labeled, fields=("cutoffs", "probcounted"), title="Total Model Probability Accounted For")
    prob_counted_fig.savefig("plots/supp_probcounted.png", bbox_inches='tight')

    invalid_pcnt_fig = generate_sup_plots_2x2(results_dict_all_labeled, fields=("cutoffs", "invalid"), title="Percentage of Templates with no Lemmas at Cut-Off")
    invalid_pcnt_fig.savefig("plots/supp_invalid-pcnt.png", bbox_inches='tight')

if __name__ == "__main__":
    print("Generating plots...")
    generate_figures()
