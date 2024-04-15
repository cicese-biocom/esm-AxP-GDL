import argparse
from pathlib import Path
from workflow.application_context import ApplicationContext
from workflow.args_parser_handler import ArgsParserHandler
from workflow.gdl_workflow import InferenceWorkflow


def inference(args):
    try:
        context = ApplicationContext(mode='inference')
        InferenceWorkflow().run_workflow(context=context, parameters=args)

    except Exception as e:
        raise


if __name__ == '__main__':
    args_handler = ArgsParserHandler()
    args = args_handler.get_eval_arguments()

    args['dataset'] = Path('datasets/TestDataset/TestDataset.csv')
    args['pdb_path'] = Path('datasets/AMPDiscover/ESMFold_pdbs/')
    args['gdl_model_path'] = Path('output_models/TestNewImplementation/Checkpoints/epoch=2_train-loss=0.70_val-loss=0'
                                  '.70.pt')
    args['output_path'] = Path('output_models/TestNewImplementation/')

    inference(args)
