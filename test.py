import logging
from src.config.types import ExecutionMode
from src.workflow.gdl_workflow import TestWorkflow
import time


def main():
    TestWorkflow(ExecutionMode.TEST).run()


if __name__ == '__main__':

    start_time = time.time()
    main()
    final_time = time.time()
    logging.getLogger('logger').info(
        f"Graph analyzer execution time in: {str(final_time - start_time)} seconds")