"""Routes experiment to specified model. Loads it.

Attributes:
    model_name_to_MPEModel_class (dict): Maps model name to MPEModel subclass.
"""


from .bert_model import BertModel
from .gpt2_model import GPT2Model
from .MPE_model import MPEModel
from .roberta_model import RobertaModel


model_name_to_MPEModel_class = {
    "bert": BertModel,
    "gpt2": GPT2Model,
    "roberta": RobertaModel,
}


def load_model(config):
    """Loads instance of model specified in config.

    Args:
        config (dict): model-level config dict with a `name` field. Is config for model initialization.

    Returns:
        MPEModel: Instance of loaded model.
    """
    if r"/" in config["name"]:
        model_name = config["name"].split("/")[-1]
    else:
        model_name = config["name"]
    model_name = model_name.split("-")[0]
    class_of_MPEModel = model_name_to_MPEModel_class.get(model_name)
    if class_of_MPEModel:
        assert issubclass(class_of_MPEModel, MPEModel)
        return class_of_MPEModel(config)
    else:
        raise ValueError(f"Unrecognized model name {model_name}.")
