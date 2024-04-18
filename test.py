from workflow.application_context import ApplicationContext
from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import TestWorkflow
import time


def test(args):
    try:
        context = ApplicationContext(mode='test')
        TestWorkflow().run_workflow(context=context, parameters=args)

    except Exception as e:
        raise


if __name__ == '__main__':
    args_handler = ArgsParserHandler()
    args = args_handler.get_eval_arguments()

    start_time = time.time()
    test(args)
    final_time = time.time()
    print(
        f"Test execution time in: {str(final_time - start_time)} seconds"
    )
