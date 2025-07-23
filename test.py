import logging
from src.config.enum import ExecutionMode
from src.workflow.gdl_workflow import TestWorkflow
from src.workflow.params_setup import argument_parser
import time


def main():
    parameters = argument_parser(ExecutionMode.TRAIN)
    TestWorkflow(parameters).run()


if __name__ == '__main__':

    start_time = time.time()
    main()
    final_time = time.time()
    logging.getLogger('logger').info(
        f"Graph analyzer execution time in: {str(final_time - start_time)} seconds")