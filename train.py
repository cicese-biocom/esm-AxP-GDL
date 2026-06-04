import logging
from src.config.types import ExecutionMode
from src.workflow.gdl_workflow import TrainingWorkflow
import time


def main():
    TrainingWorkflow(ExecutionMode.TRAIN).run()


if __name__ == '__main__':

    start_time = time.time()
    main()
    final_time = time.time()
    logging.getLogger('logger').info(
        f"Graph analyzer execution time in: {str(final_time - start_time)} seconds")