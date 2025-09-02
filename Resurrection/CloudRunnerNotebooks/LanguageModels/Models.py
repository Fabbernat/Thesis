class Gemma22bIt:
    pass


class Phi4MiniInstruct:
    pass


class Qwen25b:
    pass


class ModelNotFoundError(Exception):
    pass


def get(MODEL_NAME: str):
    match MODEL_NAME:
        case "gemma-2-2b-it":
            return Gemma22bIt()
        case "phi-4-mini-instruct":
            return Phi4MiniInstruct()
        case "qwen-2.5-b":
            return Qwen25b()
        case _:
            raise ModelNotFoundError(
                f"The given model '{MODEL_NAME}' could not be found. "
                "Make sure you don't have a typo in your model name."
            )
