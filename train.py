from pathlib import Path
from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import TrainingWorkflow
from workflow.application_context import ApplicationContext


def set_of_edge_functions(value):
    functions = value.replace(" ", "").split(',')
    functions = set(functions)
    return functions


def train(args):
    try:
        context = ApplicationContext(mode='training')
        TrainingWorkflow().run_workflow(context=context, parameters=args)

    except Exception as e:
        raise


if __name__ == '__main__':
    args_handler = ArgsParserHandler()
    args = args_handler.get_training_arguments()

    train(args)
