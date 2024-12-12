import logging

from workflow.application_context import ApplicationContext
from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import InferenceWorkflow
import time


def inference(args):
    context = ApplicationContext(mode='inference')
    InferenceWorkflow().run_workflow(context=context, parameters=args)


if __name__ == '__main__':
    try:
        args_handler = ArgsParserHandler()
        args = args_handler.get_inference_arguments()

        start_time = time.time()
        inference(args)
        final_time = time.time()
        print(
            f"Inference execution time in: {str(final_time - start_time)} seconds"
        )

    except Exception as e:
        logging.getLogger('workflow_logger').critical(e)
        quit()
