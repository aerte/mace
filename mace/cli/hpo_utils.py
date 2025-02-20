# HPO utils
# author: faerte

from mace.tools.model_script_utils import configure_model

def get_hpo_grid(args):
    if args.grid is not None:
        # Convert string representation of list to actual list
        grid_str = args.grid.strip('[]')  # Remove brackets
        grid = [float(x.strip()) for x in grid_str.split(',')]  # Split on commas and convert to float
        return grid
    else:
        return [2, 2.5, 3, 3.5, 4, 4.5, 5]
    

def configure_grid_model(args, train_loader, atomic_energies, model_foundation, heads, z_table, device):
    grid = get_hpo_grid(args)
    models, output_args_list = [], []
    for r in grid:
        args.r_max = r
        model, output_args = configure_model(args, train_loader, atomic_energies, model_foundation, heads, z_table)
        model.to(device)
        models.append(model)
        output_args_list.append(output_args)
    return models, output_args_list


def configure_hpo_model(args, train_loader, atomic_energies, model_foundation, heads, z_table, device):
    if args.hpo_type == 'grid':
        return configure_grid_model(args, train_loader, atomic_energies, model_foundation, heads, z_table, device)
    else:
        raise ValueError(f"Invalid HPO method: {args.hpo_type}")