import logging

from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import TrainingWorkflow
from workflow.application_context import ApplicationContext
import time


def train(args):
    context = ApplicationContext(mode='training')
    TrainingWorkflow().run_workflow(context=context, parameters=args)


if __name__ == '__main__':
    args_handler = ArgsParserHandler()
    args = args_handler.get_training_arguments()

    start_time = time.time()
    train(args)
    final_time = time.time()
    logging.getLogger('workflow_logger').info(
        f"Training execution time in: {str(final_time - start_time)} seconds")