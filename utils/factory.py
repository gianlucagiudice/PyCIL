from models.der import DER


def get_model(model_name, args):
    name = model_name.lower()
    if name == "der":
        return DER(args)
    else:
        assert 0
