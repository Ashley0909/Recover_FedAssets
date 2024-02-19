import pickle
from pathlib import Path

import hydra
# from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from torchvision.transforms import ToTensor, Normalize, Compose

from dataset import prepare_clientdataset
from client import generate_nnclient_fn, weighted_average
from server import get_on_fit_config, get_evaluate_fn, get_attacker_evaluate_fn
from bd_strategy import NNtrain
from model import Net, get_parameters, LeNet
from mixedbackdoor import build_testset

import torchvision.models as models
import torch.nn as nn

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    """ 1. Parse config & get experiment output dir """
    print(OmegaConf.to_yaml(cfg))
    # save_path = HydraConfig.get().runtime.output_dir

        # or print(cfg) will output the lines in a dictionary
        # change numbers by e.g. "python main.py num_clients=500"

    """ 2. Prepare dirty and clean dataset """
    # bdtrainloaders, bdvalloaders, cleantrainloaders, cleanvalloaders = prepare_nndataset(cfg.dataset_config, cfg.num_clients, cfg.batch_size)
    bdtrainloaders, bdvalloaders, cleantrainloaders, cleanvalloaders, testloaders, attackertestloaders = prepare_clientdataset(cfg.dataset_config, cfg.num_clients, cfg.batch_size, cfg.dataset)
    # testloaders = prepare_testset(cfg.num_clients, cfg.batch_size, cfg.dataset)

    """ 3. Define your clients """
    nn_client_fn = generate_nnclient_fn(cfg, cleantrainloaders, cleanvalloaders, bdtrainloaders, bdvalloaders, cfg.num_classes, cfg.num_clients, cfg.num_channels)
    # actual_client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    model = models.resnet18()
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, cfg.num_classes)

    # params = get_parameters(Net(cfg.num_classes, cfg.num_channels))
    params = get_parameters(model)

    """Start Actual Simulation"""
    nnet = fl.simulation.start_simulation(
        client_fn=nn_client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds), 
        strategy=NNtrain(
            #client's info
            fraction_fit=0.00001,                                 # fraction of clients used during training (Default to 1.0)
            min_fit_clients=cfg.num_clients_per_round_fit,        # minimum number of clients used during training (fit())
            fraction_evaluate=0.00001,                            # fraction of clients used during validation
            min_evaluate_clients=cfg.num_clients_per_round_eval,  # minimum number of clients using during validation evaluate()
            min_available_clients=cfg.num_clients,                # minimum number of total clients in the simulation
            # Remark: in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
            
            # server's side
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            on_fit_config_fn=get_on_fit_config(cfg.config_fit),  
            # on_evaluate_config_fn=evaluate_config,
            evaluate_fn=get_evaluate_fn(cfg.num_classes, cfg.num_channels, testloaders), 
            attack_evaluate_fn=get_attacker_evaluate_fn(cfg.num_classes, cfg.num_channels, attackertestloaders),
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        ),
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0, 
        }, 
    )

    # print("+++++++++++Actual Federated Learning Simulation Starts++++++++++++++")

    # """ 6. Start Actual Simulation """
    # history = fl.simulation.start_simulation(
    #     client_fn=actual_client_fn,  # a function that spawns a particular client
    #     num_clients=cfg.num_clients,  # total number of clients
    #     config=fl.server.ServerConfig(
    #         num_rounds=cfg.num_rounds
    #     ),  # minimal config for the server loop telling the number of rounds in FL
    #     strategy=strategy,  
    #     client_resources={
    #         "num_cpus": 2,
    #         "num_gpus": 0.0, 
    #     },  # (optional) controls the degree of parallelism of your simulation.
    #     # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
    #     # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.

    #     #i.e. if num_gpus is 1.0, then clients can only run one at a time, since each client needs to have access to the entire memory
    #     #i.e. if num_gpus is 0.25, then my gpu should be able to have 4 clients running concurrently
    #     # Lower resources per client allow for more clients to run concurrently
    #     # (but need to be set taking into account the compute/memory footprint of your workload)
    # )

    # """ 6. Save your results """
    # results_path = Path(save_path) / 'results.pkl'

    # results = {'history': history, 'nn': nnet}

    # # save the results as a python pickle
    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()