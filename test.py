import logging
from workflow.application_context import ApplicationContext
from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import TestWorkflow
import time


def test(args):
    context = ApplicationContext(mode='test')
    TestWorkflow().run_workflow(context=context, parameters=args)


if __name__ == '__main__':
    try:
        args_handler = ArgsParserHandler()
        args = args_handler.get_eval_arguments()

        start_time = time.time()
        test(args)
        final_time = time.time()
        logging.getLogger('workflow_logger').info(
            f"Inference execution time in: {str(final_time - start_time)} seconds")

    except Exception as e:
        logging.getLogger('workflow_logger').critical(e)
        quit()
