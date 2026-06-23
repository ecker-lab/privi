

def get_classifier(type, *args, **kwargs):
    """
    Factory function to get the classifier based on the type.
    """
    if type == "vjepa":
        from privi.crop_action_head.models.vjepa_classifier import VJEPAClassifier
        return VJEPAClassifier(*args, **kwargs)
    elif type == "vjepa2":
        from privi.crop_action_head.models.vjepa2_classifier import VJEPA2Classifier
        return VJEPA2Classifier(*args, **kwargs)
    elif type == "privi":
        from privi.crop_action_head.models.privi_classifier import PriViClassifier
        return PriViClassifier(*args, **kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {type}")



