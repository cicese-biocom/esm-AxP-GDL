import argparse
from pathlib import Path

from workflow.application_context import ApplicationContext
from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import TestWorkflow


def test(args):
    try:
        context = ApplicationContext(mode='test')
        TestWorkflow().run_workflow(context=context, parameters=args)

    except Exception as e:
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args_handler = ArgsParserHandler()
    args = args_handler.get_eval_arguments()

    test(args)
