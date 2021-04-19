"""Routes experiment to specified dataset. Loads it.

Attributes:
    dataset_name_to_MPEDataset_class (dict): Maps dataset name to MPEDataset subclass.
"""


from .custom_dataset import CustomDataset
from .MPE_dataset import MPEDataset


dataset_name_to_MPEDataset_class = {
    "custom": CustomDataset,
}


def load_dataset(config):
    """Loads instance of dataset specified in config.

    Args:
        config (dict): Dataset-level config dict with a `name` field. Is config for dataset initialization.

    Returns:
        MPEDataset: Instance of loaded dataset.
    """
    dataset_name = config["name"]
    class_of_MPEDataset = dataset_name_to_MPEDataset_class.get(dataset_name)
    if class_of_MPEDataset:
        assert issubclass(class_of_MPEDataset, MPEDataset)
        return class_of_MPEDataset(config)
    else:
        raise ValueError(f"Unrecognized dataset name {dataset_name}.")
