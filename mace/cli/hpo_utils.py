# HPO utils
# author: faerte

from mace.tools.model_script_utils import configure_model
from ax.service.ax_client import AxClient, ObjectiveProperties
from functools import partial
from mace.cli.bo_hpo import single_param_train_eval

def get_hpo_grid(args):
    if args.grid is not None:
        # Convert string representation of list to actual list
        grid_str = args.grid.strip('[]')  # Remove brackets
        grid = [float(x.strip()) for x in grid_str.split(',')]  # Split on commas and convert to float
        return grid
    else:
        return [2, 2.5, 3, 3.5, 4, 4.5, 5]
    

def configure_grid_model(args, input_dict):
    grid = get_hpo_grid(args)
    models, output_args_list = [], []
    for r in grid:
        args.r_max = r
        model, output_args = configure_model(args, input_dict['train_loader'], input_dict['atomic_energies'], input_dict['model_foundation'], input_dict['heads'], input_dict['z_table'])
        model.to(input_dict['device'])
        models.append(model)
        output_args_list.append(output_args)
    return models, output_args_list

def configure_single_param_bo(args, input_dict):
    # Create an experiment with required arguments: name, parameters, and objective_name.
    ax_client = AxClient()
    ax_client.create_experiment(
        name="tune_mace_r_max",  # The name of the experiment.
        parameters=[
            {
                "name": "r_max",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [3, 6],  # The bounds for range parameters. 
                # "values" The possible values for choice parameters .
                # "value" The fixed value for fixed parameters.
                "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, whether to use a log scale for range parameters. Defaults to False.
                # "is_ordered" Optional, a flag for choice parameters.
            }
        ],
        objectives={"loss": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
        # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
        # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
    )

    eval_func = partial(single_param_train_eval,  args=args, input_dict=input_dict)
    
    return ax_client, eval_func




def configure_hpo_model(args, input_dict):
    if args.hpo_type == 'grid':
        return configure_grid_model(args, input_dict)
    elif args.hpo_type == 'single_bo':
        return configure_single_param_bo(args, input_dict)
    else:
        raise ValueError(f"Invalid HPO method: {args.hpo_type}")