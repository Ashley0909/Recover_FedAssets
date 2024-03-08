from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar, Status, Code, EvaluateRes
from flwr.common.typing import Metrics

import torch
import flwr as fl
from model import Net, train, test, LeNet

import torch.nn as nn
import torchvision.models as models

import constant


class PresetClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_classes,
                 malicious,
                 num_channels,
                #  device,
                 ) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18()  #.cuda()
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)  #.cuda()

        self.malicious = malicious

    """receives and copies the parameter sent from server into the client's local model"""
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
        state_dict = OrderedDict({ k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    """get local model parameters and return them as a list of numpy arrays."""
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        proximal_mu = config['proximal_mu']
        poisoning_rate = config['poisoning_rate']

        # optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        # do local training
        train(self.model, self.trainloader, self.device, epochs, lr, proximal_mu, self.malicious, poisoning_rate)

        # return the updated model, the number of examples in the client, and a dictionary of metrics
        return self.get_parameters({}), len(self.trainloader), {"malicious": self.malicious}


    """client uses validation data to evaluate the model"""
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device, self.malicious, constant.P_RATE)

        return float(loss), len(self.valloader), {"accuracy": accuracy, "malicious": self.malicious}
    

#++++++++++++++++++++++++++++++++++++++++++++++Generate Client Function+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Return a function that can be used by the VirtualClientEngine to spawn a FlowerClient with client id `cid`."""
# def generate_nnclient_fn(config: DictConfig, goodtrainloaders, goodvalloaders, bdtrainloaders, bdvalloaders, num_classes, num_clients, num_channels,device):
def generate_nnclient_fn(config: DictConfig, goodtrainloaders, goodvalloaders, bdtrainloaders, bdvalloaders, num_classes, num_clients, num_channels):

    # This function will be called internally by the VirtualClientEngine
    # Each time the cid-th client is told to participate in the FL simulation (whether it is for doing fit() or evaluate())
    def client_fn(cid: str):

        modnum = int(num_clients * config.dataset_config.ratio_benign_client)  # 2 different types of clients: 60% benign and 40% malicious (num_clients is 100)

        if int(cid) < modnum:
            # Benign Client
            return PresetClient(
                trainloader=goodtrainloaders[int(cid)],
                valloader=goodvalloaders[int(cid)],
                num_classes=num_classes,
                malicious=0,
                num_channels=num_channels,
                # device=device,
            )
        else:
            # Backdoor Client 
            return PresetClient(
                trainloader=bdtrainloaders[int(cid)-modnum],
                valloader=bdvalloaders[int(cid)-modnum],
                num_classes=num_classes,
                malicious=2,
                num_channels=num_channels,
                # device=device,
            )

    # return the function to spawn client
    return client_fn


#++++++++++++++++++++++++++++++++++++++++++++++Aggregate Accuracy of a Client+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    #Aggregate and return weighted average
    return {"accuracy": sum(accuracies) / sum(examples)}