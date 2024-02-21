from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from functools import reduce
import numpy as np
from time import time

from collections import OrderedDict, Counter
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from openpyxl import load_workbook
wb = load_workbook( "CIFAR_Global.xlsx" )
# wb = load_workbook( "Clustering_per_layer.xlsx" )
ws = wb.active

import smtplib
from email.message import EmailMessage
import ssl
import constant

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

"""some helper functions so that we can convert between numpy arrays and pytorch tensors and run our code on GPU"""
USE_CUDA = torch.cuda.is_available() 
USE_MPS = torch.backends.mps.is_available()

from torch.autograd import Variable
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):
    if USE_MPS:
        return Variable(torch.tensor(v, device="mps").type(dtype).requires_grad_(requires_grad))
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

print('Using CUDA:',USE_CUDA)
print('Using MPS:', USE_MPS)

malicious_record = []
final_model = []
final_metric = []
e_olb, e_c1w, e_c2w, e_fhw, e_shw = 0, 0, 0, 0, 0
highest_accuracy_c1w, highest_accuracy_c2w, highest_accuracy_fhw, highest_accuracy_shw, highest_accuracy_olb = 1/10, 1/10, 1/10, 1/10, 1/10
lowest_accuracy_c1w, lowest_accuracy_c2w, lowest_accuracy_fhw, lowest_accuracy_shw, lowest_accuracy_olb = 0, 0, 0, 0, 0
global_targetlabel = None

class NNtrain(Strategy):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        attack_evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.attack_evaluate_fn = attack_evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn


    def __repr__(self) -> str:
        rep = f"NNtrain(accept_failures={self.accept_failures})"
        return rep


    """Return the sample size and the required number of available clients."""
    """working and unchanged"""
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    """Use a fraction of available clients for evaluation."""
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


    """Server request an initial global parameter given by a random client"""
    """unchanged but need changing"""
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    """Evaluate model parameters using an evaluation function."""
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        print("Evaluate")
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        if isinstance(parameters, list):
            parameters_ndarrays = parameters_to_ndarrays(parameters[0])
        else:
            parameters_ndarrays = parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res

        # Computing the poisoning accuracy
        attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays, {})
        if attack_eval_res is None:
            return None
        _, attack_metrics = attack_eval_res

        if server_round > 0:
            print("Global Poisoning Accuracy:", attack_metrics["accuracy"])
            ws[constant.EXCEL_CELL+str(server_round+4)] = metrics["accuracy"]
            ws[constant.EXCEL_CELL+str(server_round+315)] = attack_metrics["accuracy"]
        
        # wb.save( "MNIST_Global.xlsx" )
        wb.save( "CIFAR_Global.xlsx" )
            
        if server_round == 100:
            email_sender = '09auhoiting@gmail.com'
            email_password = 'fgkkhdwmluvpxewp'
            email_receiver = '09auhoiting@gmail.com'
            subject = 'Vscode Run Result'
            body = 'Ran Successfully. Final Accuracy is {GA}'.format(GA=metrics["accuracy"])

            em = EmailMessage()
            em['From'] = email_sender
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)

            context = ssl.create_default_context()

            with smtplib.SMTP_SSL('smtp.gmail.com',465, context=context) as smtp:
                smtp.login(email_sender, email_password)
                smtp.sendmail(email_sender, email_receiver, em.as_string())

        return loss, metrics


    """Configure the next round of training."""
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        print("Configure Fit")
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        print("server round:", server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        output = []
        if isinstance(parameters, list):
            normal_fit_ins = FitIns(parameters[0], config)
            bad_fit_ins = FitIns(parameters[1], config)

            for c in clients:
                if parameters[-1].get(c.cid) is None:
                    output.append((c, normal_fit_ins))
                elif parameters[-1].get(c.cid) == 2:
                    output.append((c, bad_fit_ins))
                elif parameters[-1].get(c.cid) == 0:
                    output.append((c, normal_fit_ins))
        else:
            fit_ins = FitIns(parameters, config)
            output = [(client, fit_ins) for client in clients]

        # Return client/config pairs
        return output


    """Configure the next round of evaluation."""
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Do not configure federated evaluation if fraction eval is 0.
        print("Configure Evaluate")
        if self.fraction_evaluate == 0.0:
            return []
    
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        if isinstance(parameters, list):
            normal_evaluate_ins = EvaluateIns(parameters[0], config)
            bad_evaluate_ins = EvaluateIns(parameters[1], config)
        else:
            evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        output = []
        if isinstance(parameters, list):
            for c in clients:
                if parameters[-1].get(c.cid) is None:
                    output.append((c, normal_evaluate_ins))
                elif parameters[-1].get(c.cid) == 2:
                    output.append((c, bad_evaluate_ins))
                elif parameters[-1].get(c.cid) == 0:
                    output.append((c, normal_evaluate_ins))
        else:
            output = [(client, evaluate_ins) for client in clients]

        # Return client/config pairs
        return output


    """Original: Aggregate fit results using weighted average."""
    """Changed: Collect and combine all the results as a labelled data set"""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        global malicious_record, highest_accuracy_c1w, lowest_accuracy_c1w, highest_accuracy_c2w, lowest_accuracy_c2w, highest_accuracy_fhw, lowest_accuracy_fhw, highest_accuracy_shw, lowest_accuracy_shw, highest_accuracy_olb, lowest_accuracy_olb, final_model, final_metric, global_targetlabel, e_olb, e_c1w, e_c2w, e_fhw, e_shw

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        """That procedure makes sure the known malicious clients are not considered"""
        weights_results = []
        all_id = []
        malicious = []
        new_results = []
        individual_acc = []

        evil_numexamples = []
        evil_parameter = []
        i = 0
        total = 0
        count = 0
        for cp, fit_res in results:
            if cp.cid in malicious_record:
                evil_numexamples.append(fit_res.num_examples)
                evil_parameter.append(parameters_to_ndarrays(fit_res.parameters))
                # Computing individual poisoning accuracies
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays, {})
                if attack_eval_res is None:
                    return None
                _, metrics = attack_eval_res
                total +=  metrics["accuracy"]
                count += 1
            else:
                all_id.append((i, cp.cid))
                malicious.append(str(fit_res.metrics["malicious"]))
                weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                new_results.append((cp, fit_res))
                # Computing individual accuracies
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
                if eval_res is None:
                    return None
                _, metrics = eval_res
                individual_acc.append(metrics["accuracy"])
                i += 1

        print("individual accuracies", individual_acc)
        print("malicious is", malicious)
        
        num_examples = [res[1] for res in weights_results]
        print("Number of Evil Clients:", len(evil_numexamples))

        if len(num_examples) == 0:
            print("Only Evil Clients in this round, void this round.")
            return final_model, final_metric
        
        client_id = np.array(all_id)[:,1]
        local_cid = np.array(all_id)[:,0]
        parameter = [client[0] for client in weights_results]  #[c1:[10array], c2:[10array], ...]  #client[1] is the number of examples

        print("Number of Preset Malicious Clients is", malicious.count("2"))
        print("Number of Preset Benign Clients is", malicious.count("0"))
        # print(len(parameter[0][0]))  #6
        # print(len(parameter[0][1]))  #6
        # print(len(parameter[0][2]))  #16
        # print(len(parameter[0][3]))  #16
        # print(len(parameter[0][4]))  #120
        # print(len(parameter[0][5]))  #120
        # print(len(parameter[0][6]))  #84
        # print(len(parameter[0][7]))  #84
        # print(len(parameter[0][8]))  #10     #contains the weight values each output neuron gets (each neuron should receive 84 weights)
        # print(len(parameter[0][9]))  #10     #contains the bias value of the 10 output neurons

        print("Here, what is the size of the fully connected layer?", len(parameter[0][-1]))
        print("How about the second last layer?", len(parameter[0][-2]))

        fcb = [sublist[-1] for sublist in parameter]
        fcw = []
        for i in range(len(parameter)): 
            w = 0
            vector = []  #set up a vector for each client
            for j in range(len(parameter[i][-2])): #10
                w = np.sum(parameter[i][-2][j]) #84
                vector.append(w)
            fcw.append(np.array(vector))

        textstr = ''
        for l, g in enumerate(local_cid):
            textstr += f'Client {g} => {int(malicious[l])} \n'

        plt.imshow(np.array(fcb), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.text(-12, 12, textstr, fontsize=8, verticalalignment='center', horizontalalignment='left')
        plt.xlabel("FC Layer Bias")
        plt.ylabel("Clients")
        plt.title("Heatmap of all clients' FC Layer Bias")
        plt.savefig('./FCB.png')
        plt.close()

        plt.imshow(np.array(fcw), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.text(-12, 12, textstr, fontsize=8, verticalalignment='center', horizontalalignment='left')
        plt.xlabel("FC Layer Weights")
        plt.ylabel("Clients")
        plt.title("Heatmap of all clients' FC Layer Weights")
        plt.savefig('./FCW.png')
        plt.close()

        _, _, _, _ = nd_clustering(parameter, client_id, malicious, fcb, "fcb", server_round, individual_acc, e=0)

        _, _, _, _ = nd_clustering(parameter, client_id, malicious, fcw, "fcw", server_round, individual_acc, e=0)

        """Check backdoor task accuracy"""
        for x in range(len(new_results)):
            if malicious[x] == '2':
                _, fit_res = new_results[x]
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays, {})
                if attack_eval_res is None:
                    return None
                _, metrics = attack_eval_res
                total +=  metrics["accuracy"]

        if malicious.count("2") != 0 or count > 0:
            poisoning_acc = total/(malicious.count("2")+count)
            print("Average poisoning accuracy:", poisoning_acc)
        else:
            poisoning_acc = "N/A"
            print("Average poisoning accuracy: N/A")

        """Implement n-dimension clustering"""
        biases = []
        weights = []
        first_hidden_b = []
        first_hidden_w = []
        second_hidden_b = []
        second_hidden_w = []
        conv1_w = []
        conv1_b = []
        conv2_w = []
        conv2_b = []
        for i in range(len(parameter)): 
            conv1_b.append(parameter[i][1])
            conv2_b.append(parameter[i][3])
            first_hidden_b.append(parameter[i][5])
            second_hidden_b.append(parameter[i][7])
            biases.append(parameter[i][9])

            w = 0
            vector = []  #set up a vector for each client
            for j in range(len(parameter[i][8])): #10
                w = np.sum(parameter[i][8][j]) #84
                vector.append(w)
            weights.append(np.array(vector))

            w = 0
            sh_vector = []
            for k in range(len(parameter[i][6])):
                w = np.sum(parameter[i][6][k])
                sh_vector.append(w)
            second_hidden_w.append(np.array(sh_vector))

            w = 0
            fh_vector = []
            for z in range(len(parameter[i][4])):
                w = np.sum(parameter[i][4][z])
                fh_vector.append(w)
            first_hidden_w.append(np.array(fh_vector))

            w = 0
            conv2_vector = []
            for z in range(len(parameter[i][2])):
                w = np.sum(parameter[i][2][z])
                conv2_vector.append(w)
            conv2_w.append(np.array(conv2_vector))

            w = 0
            conv1_vector = []
            for z in range(len(parameter[i][0])):
                w = np.sum(parameter[i][0][z])
                conv1_vector.append(w)
            conv1_w.append(np.array(conv1_vector))
        
        """Find Layer parameter of evil clients"""
        evil_conv1_w = []
        evil_conv1_b = []
        evil_conv2_w = []
        evil_conv2_b = []
        evil_biases = []
        evil_weights = []
        evil_fh_b = []
        evil_fh_w = []
        evil_sh_b = []
        evil_sh_w = []
        if len(evil_parameter) > 0:
            for i in range(len(evil_parameter)):
                evil_biases.append(evil_parameter[i][9])
                evil_sh_b.append(evil_parameter[i][7])
                evil_fh_b.append(evil_parameter[i][5])
                evil_conv2_b.append(evil_parameter[i][3])
                evil_conv1_b.append(evil_parameter[i][1])
                vector = []  #set up a vector for each client
                w = 0
                for j in range(len(evil_parameter[i][8])): #10
                    w = np.sum(evil_parameter[i][8][j]) #84
                    vector.append(w)
                evil_weights.append(np.array(vector))

                w = 0
                sh_vector = []
                for k in range(len(evil_parameter[i][6])):
                    w = np.sum(evil_parameter[i][6][k])
                    sh_vector.append(w)
                evil_sh_w.append(np.array(sh_vector))

                w = 0
                fh_vector = []
                for z in range(len(evil_parameter[i][4])):
                    w = np.sum(evil_parameter[i][4][z])
                    fh_vector.append(w)
                evil_fh_w.append(np.array(fh_vector))

                w = 0
                conv2_vector = []
                for z in range(len(evil_parameter[i][2])): 
                    w = np.sum(evil_parameter[i][2][z])
                    conv2_vector.append(w)
                evil_conv2_w.append(np.array(conv2_vector))

                w = 0
                conv1_vector = []
                for z in range(len(evil_parameter[i][0])):
                    w = np.sum(evil_parameter[i][0][z])
                    conv1_vector.append(w)
                evil_conv1_w.append(np.array(conv1_vector))

        # print("eolb, e1w, ec2w, cfhw, eshw = ", e_olb, e_c1w, e_c2w, e_fhw, e_shw)
        
        comb_C_olb, record_olb, acc_diff_olb, highest_accuracy_olb, lowest_accuracy_olb, e_olb = full_clustering(parameter, client_id, malicious, biases, "biases", server_round, individual_acc, e_olb, highest_accuracy_olb, lowest_accuracy_olb)
        comb_C_c1w, record_c1w, acc_diff_c1w, highest_accuracy_c1w, lowest_accuracy_c1w, e_c1w = full_clustering(parameter, client_id, malicious, conv1_w, "conv1w", server_round, individual_acc, e_c1w, highest_accuracy_c1w, lowest_accuracy_c1w)
        comb_C_c2w, record_c2w, acc_diff_c2w, highest_accuracy_c2w, lowest_accuracy_c2w, e_c2w = full_clustering(parameter, client_id, malicious, conv2_w, "conv2w", server_round, individual_acc, e_c2w, highest_accuracy_c2w, lowest_accuracy_c2w)
        comb_C_fhw, record_fhw, acc_diff_fhw, highest_accuracy_fhw, lowest_accuracy_fhw, e_fhw = full_clustering(parameter, client_id, malicious, first_hidden_w, "fhw", server_round, individual_acc, e_fhw, highest_accuracy_fhw, lowest_accuracy_fhw)
        comb_C_shw, record_shw, acc_diff_shw, highest_accuracy_shw, lowest_accuracy_shw, e_shw = full_clustering(parameter, client_id, malicious, second_hidden_w, "shw", server_round, individual_acc, e_shw, highest_accuracy_shw, lowest_accuracy_shw)

        """CIFAR-10"""
        # if server_round <= 10:
        #     comb_C = comb_C_fhw
        #     record = record_fhw
        #     acc_diff = acc_diff_fhw
        # elif server_round <= 50:
        #     comb_C = comb_C_c1w
        #     record = record_c1w
        #     acc_diff = acc_diff_c1w
        # else:
        #     comb_C = comb_C_shw
        #     record = record_shw
        #     acc_diff = acc_diff_shw

        """MNIST or CIFAR NonIID"""
        comb_C = comb_C_olb
        record = record_olb
        acc_diff = acc_diff_olb

        bad_num_examples = np.array(num_examples)[comb_C == 2]
        good_num_examples = np.array(num_examples)[comb_C == 0]
        bad_clients = local_cid[comb_C == 2]
        good_clients = local_cid[comb_C == 0]

        correct = 0
        for i in range(len(comb_C)):
            if (comb_C[i] == 0) and (malicious[i] == "0"):
                correct += 1
            elif (comb_C[i] == 2) and (malicious[i] == "2"):
                correct += 1
        clustering_acc = correct / len(malicious)

        print("Final Clustering acc is", clustering_acc)

        """Assume Clustering 100%"""
        # bad_index = [index for index,value in enumerate(malicious) if value == "2"]
        # good_index = [index for index,value in enumerate(malicious) if value == "0"]
        # bad_num_examples = np.array(num_examples)[bad_index]
        # good_num_examples = np.array(num_examples)[good_index]
        # bad_clients = local_cid[bad_index]
        # good_clients = local_cid[good_index]

        # print("length of bad clients:", len(bad_clients))
        # print("length of good clients:", len(good_clients))

        if record == 1:
            global_bad = client_id[comb_C == 2]
            # global_bad = client_id[bad_index]
            malicious_record.extend(global_bad)
            malicious_record = list(set(malicious_record)) #avoid duplicates

        print(len(malicious_record))

        determined_status = {}
        for i in range(len(weights_results)):
            if local_cid[i] in good_clients:
                determined_status[client_id[i]] = 0
            elif local_cid[i] in bad_clients:
                determined_status[client_id[i]] = 2

        ws[constant.EXCEL_CELL+str(server_round+107)] = clustering_acc
        ws[constant.EXCEL_CELL+str(server_round+211)] = poisoning_acc
                
        good_weights, good_biases, good_fh_w, good_fh_b, good_sh_w, good_sh_b, good_conv1_w, good_conv1_b, good_conv2_w, good_conv2_b = generateparams(weights, biases, first_hidden_w, first_hidden_b, second_hidden_w, second_hidden_b, conv1_w, conv1_b, conv2_w, conv2_b, comb_C, True)
        bad_weights, bad_biases, bad_fh_w, bad_fh_b, bad_sh_w, bad_sh_b, bad_conv1_w, bad_conv1_b, bad_conv2_w, bad_conv2_b = generateparams(weights, biases, first_hidden_w, first_hidden_b, second_hidden_w, second_hidden_b, conv1_w, conv1_b, conv2_w, conv2_b, comb_C, False)

        # good_weights, good_biases, good_fh_w, good_fh_b, good_sh_w, good_sh_b, good_conv1_w, good_conv1_b, good_conv2_w, good_conv2_b = generateparams_nocluster(weights, biases, first_hidden_w, first_hidden_b, second_hidden_w, second_hidden_b, conv1_w, conv1_b, conv2_w, conv2_b, malicious, True)
        # bad_weights, bad_biases, bad_fh_w, bad_fh_b, bad_sh_w, bad_sh_b, bad_conv1_w, bad_conv1_b, bad_conv2_w, bad_conv2_b = generateparams_nocluster(weights, biases, first_hidden_w, first_hidden_b, second_hidden_w, second_hidden_b, conv1_w, conv1_b, conv2_w, conv2_b, malicious, False)
        
        # show_clustered_plots(good_clients, bad_clients, 
        #         good_conv1_w, bad_conv1_w, good_conv1_b, bad_conv1_b, 
        #         good_conv2_w, bad_conv2_w, good_conv2_b, bad_conv2_b,
        #        good_fh_w, bad_fh_w, good_fh_b, bad_fh_b,
        #        good_sh_w, bad_sh_w, good_sh_b, bad_sh_b,
        #        good_weights, bad_weights, good_biases, bad_biases, server_round)

        """Detecting Target Label"""
        if (len(good_clients) > 0) and (len(bad_clients) > 0 or len(evil_numexamples) > 0):
            dist_list = []
            for i in range(len(good_biases[0])):
                good_biases_average = compute_average(good_biases[:,i], len(good_clients))
                if len(bad_clients) == 0:
                    bad_biases_average = compute_average(np.array(evil_biases)[:,i], len(evil_numexamples))
                elif len(evil_numexamples) == 0:
                    bad_biases_average = compute_average(bad_biases[:,i], len(bad_clients))
                else:
                    bad_biases_average = compute_average(np.concatenate((bad_biases, evil_biases), axis=0)[:,i], (len(bad_clients)+len(evil_numexamples)))

                dist = abs(good_biases_average - bad_biases_average)
                dist_list.append(dist)

            target_label = np.argmax(np.array(dist_list))
            global_targetlabel = target_label
        elif len(good_clients) == 0 and global_targetlabel != None:
            target_label = global_targetlabel
        elif len(bad_clients) == 0 and len(evil_numexamples) == 0 and global_targetlabel != None:
            target_label = global_targetlabel
        else:
            target_label = None

        print("Target label is", target_label)

        """After detecting the clients and their target label, make a function that determines the weight of contribution"""
        final_aggregated = []
        bad_model = []

        # print("Convolutional Layers")
        c1w_aggregated, bad_c1w = aggregate_weights("conv", good_conv1_w, good_num_examples, bad_conv1_w, bad_num_examples, parameter, good_clients, bad_clients, 0, False, None, evil_conv1_w, evil_parameter, evil_numexamples, server_round, acc_diff)

        # if c1w_aggregated == []:  # if theres no good clients ALL BENIGN
        #     print("No good clients")
        #     return final_model, final_metric
        
        c1b_aggregated, bad_c1b = aggregate_biases("conv", good_conv1_b, good_num_examples, bad_conv1_b, bad_num_examples, False, None, evil_conv1_b, evil_numexamples, server_round, acc_diff)
        c2w_aggregated, bad_c2w = aggregate_weights("conv", good_conv2_w, good_num_examples, bad_conv2_w, bad_num_examples, parameter, good_clients, bad_clients, 2, False, None, evil_conv2_w, evil_parameter, evil_numexamples, server_round, acc_diff)
        c2b_aggregated, bad_c2b = aggregate_biases("conv", good_conv2_b, good_num_examples, bad_conv2_b, bad_num_examples, False, None, evil_conv2_b, evil_numexamples, server_round, acc_diff)
        # print("First Hidden Layer")
        fhw_aggregated, bad_fhw = aggregate_weights("fh", good_fh_w, good_num_examples, bad_fh_w, bad_num_examples, parameter, good_clients, bad_clients, 4, False, None, evil_fh_w, evil_parameter, evil_numexamples, server_round, acc_diff)
        fhb_aggregated, bad_fhb = aggregate_biases("fh", good_fh_b, good_num_examples, bad_fh_b, bad_num_examples, False, None, evil_fh_b, evil_numexamples, server_round, acc_diff)
        # print("Second Hidden Layer")
        shw_aggregated, bad_shw = aggregate_weights("sh", good_sh_w, good_num_examples, bad_sh_w, bad_num_examples, parameter, good_clients, bad_clients, 6, False, None, evil_sh_w, evil_parameter, evil_numexamples, server_round, acc_diff)
        shb_aggregated, bad_shb = aggregate_biases("sh", good_sh_b, good_num_examples, bad_sh_b, bad_num_examples, False, None, evil_sh_b, evil_numexamples, server_round, acc_diff)
        # print("Output Layer")
        weight_aggregated, bad_weight = aggregate_weights("ol", good_weights, good_num_examples, bad_weights, bad_num_examples, parameter, good_clients, bad_clients, 8, True, target_label, evil_weights, evil_parameter, evil_numexamples, server_round, acc_diff)
        bias_aggregated, bad_bias = aggregate_biases("ol", good_biases, good_num_examples, bad_biases, bad_num_examples, True, target_label, evil_biases, evil_numexamples, server_round, acc_diff)

        final_aggregated.append(c1w_aggregated)
        final_aggregated.append(c1b_aggregated)
        final_aggregated.append(c2w_aggregated)
        final_aggregated.append(c2b_aggregated)
        final_aggregated.append(fhw_aggregated)
        final_aggregated.append(fhb_aggregated)
        final_aggregated.append(shw_aggregated)
        final_aggregated.append(shb_aggregated)
        final_aggregated.append(weight_aggregated)
        final_aggregated.append(bias_aggregated)
        # parameters_good = ndarrays_to_parameters(final_aggregated)
        parameters_aggregated = ndarrays_to_parameters(final_aggregated)

        bad_model.append(bad_c1w)
        bad_model.append(bad_c1b)
        bad_model.append(bad_c2w)
        bad_model.append(bad_c2b)
        bad_model.append(bad_fhw)
        bad_model.append(bad_fhb)
        bad_model.append(bad_shw)
        bad_model.append(bad_shb)
        bad_model.append(bad_weight)
        bad_model.append(bad_bias)
        # parameters_bad = ndarrays_to_parameters(bad_model)

        # parameters_aggregated = []
        # parameters_aggregated.append(parameters_good)
        # parameters_aggregated.append(parameters_bad)
        # parameters_aggregated.append(determined_status)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Record thie final model in case if the next round is a void round
        final_model = parameters_aggregated
        final_metric = metrics_aggregated

        return parameters_aggregated, metrics_aggregated


    """Aggregate evaluation losses using weighted average."""
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        print("Aggregate evaluate")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


def nd_clustering(parameter, cid, malicious, layer, name, server_round, indi_acc, e):
    label = []
    textstr = ''
    for k in range(len(parameter)):
        label.append("Client" + str(cid[k]) + "=>" + malicious[k] + "\n")
        textstr += f'Client {str(cid[k])} => {malicious[k]} \n'
    
    if len(parameter) < 2:
        return [0], [0], 0
        
    """Run PCA on the n dimensional data"""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(layer)

    """Non-IID"""
    e = 0.05
    mp = 5
    e -= 0.0025*(server_round/5)
    mp -= (server_round//20) 
    db = DBSCAN(eps=max(e, 0.03), min_samples=max(3,mp)).fit(reduced_data)  #original (exp=1.1, min_samples=5)
    comb_C = db.labels_
    
    """IID"""
    # if server_round < 6:
    #     kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4).fit(reduced_data)
    #     comb_C = kmeans.predict(reduced_data)

    #     centroids = kmeans.cluster_centers_
    #     centroids_assignment = kmeans.predict(centroids)
    #     unique_labels = np.unique(comb_C)
    #     max_dist = []
    #     for l in unique_labels:
    #         distances = euclidean_distances(reduced_data[comb_C == l], centroids[centroids_assignment == l])
    #         max_dist.append(np.max(distances))
    #     e = np.max(max_dist)
    # else:
    #     mp = 5
    #     e -= 0.0025*(server_round/5)
    #     mp -= (server_round//20) 
    #     # print("e is", e)
    #     # print("mp is", mp)
    #     db = DBSCAN(eps=max(e, 0.03), min_samples=max(3,mp)).fit(reduced_data)
    #     comb_C = db.labels_

    # Plotting the clusters
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='2d')
    plt.figure(figsize=(8, 6))

    # Assigning colors to clusters
    unique_labels = np.unique(comb_C)
    colors = plt.cm.bwr(np.linspace(0, 1, len(unique_labels)))

    centroids = []
    for l, color in zip(unique_labels, colors):
        if l == -1:  # Outliers are labeled as -1
            color = 'gray'
        class_member_mask = (comb_C == l)
        xy = reduced_data[class_member_mask]
        cluster_points = reduced_data[comb_C == l]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], edgecolors='k', s=50, label='Cluster {}'.format(l))

    plt.title("DBSCAN Clustering {0} of {1} clients".format(name, len(parameter)))
    plt.legend()
    
    comb0 = np.array(reduced_data)[comb_C == 0]
    comb1 = np.array(reduced_data)[comb_C != 0]
    clabel0 = np.array(label)[comb_C == 0]
    clabel1 = np.array(label)[comb_C != 0]
    texts = []
    num = 1
    for i, txt in enumerate(clabel0):
        # texts.append(ax.text(comb0[i][0], comb0[i][1], comb0[i][2], txt))
        texts.append(plt.text(comb0[i][0], comb0[i][1], txt))
        num *= -1
    for i, txt in enumerate(clabel1):
        # texts.append(ax.text(comb1[i][0], comb1[i][1], comb1[i][2], txt))
        texts.append(plt.text(comb1[i][0], comb1[i][1], txt))
        num *= -1

    plt.savefig('./{0}, Round {1}.png'.format(name, server_round))
    plt.close()

    reps = []
    for l in unique_labels:
        print("l is", l)
        cluster = np.array(indi_acc)[comb_C == l]
        median = np.median(cluster) 
        reps.append((median, l))

    """Set the largest cluster to be benign as a default"""
    counts = Counter(comb_C)
    benign_class = max(counts, key=counts.get)

    return comb_C, reps, benign_class, e

def show_clustered_plots(good_clients, bad_clients, 
                good_c1w, bad_c1w, good_c1b, bad_c1b, 
                good_c2w, bad_c2w, good_c2b, bad_c2b,
               good_fh_w, bad_fh_w, good_fh_b, bad_fh_b,
               good_sh_w, bad_sh_w, good_sh_b, bad_sh_b,
               good_weights, bad_weights, good_biases, bad_biases, server_round):
    textstr = ''
    for i in range(len(good_clients)):
        textstr += f'Client {str(good_clients[i])} => 0 \n'

    for i in range(len(bad_clients)):
        textstr += f'Client {str(bad_clients[i])} => 2 \n'

    """Plot the heat map of the first conv layer's weight values"""
    plt.imshow(np.concatenate((good_c1w, bad_c1w), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Convolutional Layer 1 Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Convolutional Layer 1's weights")
    plt.savefig('Heatmaps_1/C1W Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the first conv layer's bias vlaues"""
    plt.imshow(np.concatenate((good_c1b, bad_c1b), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Convolutional Layer 1 Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Convolutional Layer 1's biases")
    plt.savefig('Heatmaps_1/C1B Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the second conv layer's weight values"""
    plt.imshow(np.concatenate((good_c2w, bad_c2w), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Convolutional Layer 2 Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Convolutional Layer 2's weights")
    plt.savefig('Heatmaps_1/C2W Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the second conv layer's bias vlaues"""
    plt.imshow(np.concatenate((good_c2b, bad_c2b), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Convolutional Layer 2 Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Convolutional Layer 2's biases")
    plt.savefig('Heatmaps_1/C2B Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the first hidden layer's weight values"""
    plt.imshow(np.concatenate((good_fh_w, bad_fh_w), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("First Hidden Layer Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' First Hidden Layer's weights")
    plt.savefig('Heatmaps_1/FHW Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the first hidden layer's bias vlaues"""
    plt.imshow(np.concatenate((good_fh_b, bad_fh_b), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("First Hidden Layer Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' First Hidden Layer's biases")
    plt.savefig('Heatmaps_1/FHB Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the second hidden layer's weight values"""
    plt.imshow(np.concatenate((good_sh_w, bad_sh_w), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Second Hidden Layer Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Second Hidden Layer's weights")
    plt.savefig('Heatmaps_1/SHW Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the second hidden layer's bias vlaues"""
    plt.imshow(np.concatenate((good_sh_b, bad_sh_b), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Second Hidden Layer Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Second Hidden Layer's biases")
    plt.savefig('Heatmaps_1/SHB Round {0}.png'.format(server_round))
    plt.close()

    # """Plot the heat map of the third hidden layer's weight values"""
    # plt.imshow(np.concatenate((good_th_w, bad_th_w), axis=0), cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    # plt.xlabel("Third Hidden Layer Weights")
    # plt.ylabel("Clients")
    # plt.title("Heatmap of all clients' Third Hidden layer's weights")
    # plt.close()
    # plt.show(block=True)

    # """Plot the heat map of the third hidden layer's bias vlaues"""
    # plt.imshow(np.concatenate((good_th_b, bad_th_b), axis=0), cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    # plt.xlabel("Third Hidden Layer Biases")
    # plt.ylabel("Clients")
    # plt.title("Heatmap of all clients' Third Hidden layer's biases")
    # plt.close()
    # plt.show(block=True)

    """Plot the heat map of the last layer's weight values"""
    plt.imshow(np.concatenate((good_weights, bad_weights), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Last Layer Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' last layer's weights")
    plt.savefig('Heatmaps_1/OLW Round {0}.png'.format(server_round))
    plt.close()

    """Plot the heat map of the last layer's bias vlaues"""
    plt.imshow(np.concatenate((good_biases, bad_biases), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Last Layer Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' last layer's biases")
    plt.savefig('Heatmaps_1/OLB Round {0}.png'.format(server_round))
    plt.close()

def show_plots(local_cid, malicious,
               first_hidden_w, first_hidden_b,
               second_hidden_w, second_hidden_b,
               weights, biases):
    textstr = ''
    for l, g in enumerate(local_cid):
        textstr += f'Client {g} => {int(malicious[l])} \n'

    """Plot the heat map of the first hidden layer's weight values"""
    plt.imshow(first_hidden_w, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("First Hidden Layer Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' First Hidden Layer's weights")
    # plt.show(block=True)

    """Plot the heat map of the first hidden layer's bias vlaues"""
    plt.imshow(first_hidden_b, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-12, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("First Hidden Layer Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' First Hidden Layer's biases")
    # plt.show(block=True)

    """Plot the heat map of the second hidden layer's weight values"""
    plt.imshow(second_hidden_w, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Second Hidden Layer Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Second Hidden Layer's weights")
    # plt.show(block=True)

    """Plot the heat map of the second hidden layer's bias vlaues"""
    plt.imshow(second_hidden_b, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Second Hidden Layer Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' Second Hidden Layer's biases")
    # plt.show(block=True)

    """Plot the heat map of the last layer's weight values"""
    plt.imshow(weights, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Last Layer Weights")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' last layer's weights")
    # plt.show(block=True)

    print("printing last layers' biases")
    """Plot the heat map of the last layer's bias vlaues"""
    plt.imshow(biases, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.text(-8, 12, textstr, fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.xlabel("Last Layer Biases")
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' last layer's biases")
    # plt.show(block=True)
    plt.close()

def compute_average(data, count):
    average = np.sum(data, axis=0) / count

    return average

def naive_aggregate(parameters: List[NDArrays], num_examples: List[int]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    if len(num_examples) == 0:
        return []
    
    num_examples_total = sum(num_examples)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = []
    for i in range(len(num_examples)):
        weighted_weights.append(parameters[i] * num_examples[i])

    # Compute average weights of each layer
    if len(np.shape(weighted_weights)) == 1:
        weights_prime: NDArrays = [np.sum(weighted_weights) / num_examples_total]
    else:
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
    
    return weights_prime

def aggregate_biases(name, good_layer, good_num_examples, bad_layer, bad_num_examples, last: bool, target_label: int, evil_layer, evil_num_examples, server_round, acc_diff):
    # print("Aggregate biases")
    aggregate_for_good = []

    if evil_layer != []:
        evil_aggregated = naive_aggregate(evil_layer, evil_num_examples)
    else:
        evil_aggregated = []

    if acc_diff != 0: # if there is no good clients
        # print("acc_diff", acc_diff)
        bad_aggregated = naive_aggregate(bad_layer, bad_num_examples)
        for j in range(len(bad_layer[0])):
            sim = np.exp(constant.EVIL_LAMBDA * acc_diff)
            evil_sim = np.exp(constant.EVIL_LAMBDA * acc_diff)
            # print("sim")
            # if sim > 0.1:
            #     print(sim)
            # print("evil sim")
            # if evil_sim > 0.1:
            #     print(evil_sim)
            if evil_aggregated != []:
                weighted_param_aggregated = ((sim * bad_aggregated[j]) + (evil_sim * evil_aggregated[j]))/(evil_sim + sim)
            else:
                weighted_param_aggregated = sim * bad_aggregated[j]
            aggregate_for_good.append(weighted_param_aggregated)

        if name == "conv":
            return [], []
        else:
            return aggregate_for_good, []
        # return [], []  # All Benign
    
    good_aggregated = naive_aggregate(good_layer, good_num_examples)
    if len(bad_layer) == 0:  # no bad clients
        aggregated_result = []
        if evil_aggregated != []: 
            for j in range(len(good_layer[0])):
                evil_dist = abs(good_aggregated[j] - evil_aggregated[j])
                evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)

                # print("evil sim")
                # if evil_sim_weight > 0.1:
                #     print(evil_sim_weight)
                weighted_param_aggregated = (good_aggregated[j] + (evil_sim_weight * evil_aggregated[j]))/(1+evil_sim_weight)
                aggregated_result.append(weighted_param_aggregated)
        else:
            aggregated_result = good_aggregated

        if name == "conv":
            return good_aggregated, good_aggregated
        else:
            return aggregated_result, good_aggregated
        # return good_aggregated, good_aggregated  # All Benign
    
    bad_aggregated = naive_aggregate(bad_layer, bad_num_examples)

    for j in range(len(good_layer[0])):
        if evil_aggregated != []:
            evil_dist = abs(good_aggregated[j] - evil_aggregated[j])
            evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
            # evil_sim_weight = 1  # equally
        else:
            evil_sim_weight = 0
             
        if last and j == target_label:
            sim_weight = 0
            evil_sim_weight = 0
        else:
            dist = abs(good_aggregated[j] - bad_aggregated[j])
            sim_weight = np.exp(constant.MALI_LAMBDA * dist)  #Updated similarity weight CHANGED 5.5
        # sim_weight = 1  # equally
            
        # print("Sim")
        # if sim_weight > 0.1:
        #     print(sim_weight)
        # print("evil sim")
        # if evil_sim_weight > 0.1:
        #     print(evil_sim_weight)

        if evil_sim_weight == 0:
            weighted_param_aggregated = (good_aggregated[j] + (sim_weight * bad_aggregated[j])) / (1 + sim_weight)
        else:
            weighted_param_aggregated = (good_aggregated[j] + (sim_weight * bad_aggregated[j]) + (evil_sim_weight * evil_aggregated[j])) / (1 + sim_weight + evil_sim_weight)
        aggregate_for_good.append(weighted_param_aggregated)

    if name == "conv":
        return good_aggregated, good_aggregated
    else:
        return aggregate_for_good, good_aggregated
    # return good_aggregated, good_aggregated  # All Benign

def aggregate_weights(name, good_layer, good_num_examples, bad_layer, bad_num_examples, parameter, good_client, bad_client, k, last: bool, target_label: int, evil_layer, evil_parameter, evil_num_examples, server_round, acc_diff):
    # print("Aggregate weights")
    aggregate_for_good = []
    good_weighted_weights = []
    bad_weighted_weights = []
    evil_weighted_weights = []
    total_evil_num = sum(evil_num_examples)
    total_bad_num = sum(bad_num_examples)
    total_good_num = sum(good_num_examples)

    # Aggregate all evil clients' original expanded weights
    if total_evil_num != 0:
        for i in range(len(evil_parameter)): # number of evil clients
            evil_weighted_weights.append(evil_parameter[i][k] * evil_num_examples[i])
        evil_weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total_evil_num
            for layer_updates in zip(*evil_weighted_weights)
        ]
        evil_aggregated = naive_aggregate(evil_layer, evil_num_examples)
    else:
        evil_aggregated = []

    if acc_diff != 0: # if there is no good clients
        for j, i in enumerate(bad_client): #each malicious client
            bad_weighted_weights.append(parameter[int(i)][k] * bad_num_examples[j])
        bad_weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / total_bad_num
            for layer_updates in zip(*bad_weighted_weights)
        ]

        for j in range(len(parameter[0][k])): #each neuron
            sim = np.exp(constant.EVIL_LAMBDA * acc_diff)
            evil_sim = np.exp(constant.EVIL_LAMBDA * acc_diff)
            # print("sim")
            # if sim > 0.1:
            #     print(sim)
            # print("Evil sim")
            # if evil_sim > 0.1:
            #     print(evil_sim)

            if evil_aggregated != []:
                weighted_param_aggregated = ((sim * bad_weights_prime[j]) + (evil_sim * evil_weights_prime[j]))/(sim + evil_sim)
            else:
                weighted_param_aggregated = sim * bad_weights_prime[j]
            aggregate_for_good.append(weighted_param_aggregated)

        if name == "conv":
            return [], []
        else:
            return aggregate_for_good, []
        # return [], []  # All Benign

    # Aggregate all good clients' original expanded weights
    good_aggregated = naive_aggregate(good_layer, good_num_examples)
    for j, i in enumerate(good_client): #each benign client
        good_weighted_weights.append(parameter[int(i)][k] * good_num_examples[j])
    good_weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_good_num
        for layer_updates in zip(*good_weighted_weights)
    ]

    # print("Total bad num is", total_bad_num)

    if total_bad_num == 0: # no bad client
        aggregated_result = []
        if evil_aggregated != []:
            for j in range(len(parameter[0][k])):
                evil_dist = abs(good_aggregated[j] - evil_aggregated[j])
                evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
                # print("evil sim")
                # if evil_sim_weight > 0.1:
                #     print(evil_sim_weight)
                weighted_param_aggregated = (good_weights_prime[j] + (evil_sim_weight * evil_weights_prime[j]))/(1+evil_sim_weight)
                aggregated_result.append(weighted_param_aggregated)
        else:
            aggregated_result = good_weights_prime
        
        if name == "conv":
            return good_weights_prime, good_weights_prime
        else:
            return aggregated_result, good_weights_prime
        # return good_weights_prime, good_weights_prime   # All Benign
    
    # Aggregate all bad clients' original expanded weights
    bad_aggregated = naive_aggregate(bad_layer, bad_num_examples)
    for j, i in enumerate(bad_client): #each malicious client
        bad_weighted_weights.append(parameter[int(i)][k] * bad_num_examples[j])
    bad_weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_bad_num
        for layer_updates in zip(*bad_weighted_weights)
    ]

    for j in range(len(parameter[0][k])): #each neuron
        if evil_aggregated != []:
            evil_dist = abs(good_aggregated[j] - evil_aggregated[j])
            evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
            # evil_sim_weight = 1
        else:
            evil_sim_weight = 0

        if last and j == target_label:
            sim_weight = 0
            evil_sim_weight = 0 
        else:
            dist = abs(good_aggregated[j] - bad_aggregated[j])
            sim_weight = np.exp(constant.MALI_LAMBDA * dist)  #Updated similarity weight CHANGED 7
        
        # print("Sim weight")
        # if sim_weight > 0.1:
        #     print(sim_weight)
        # print("Evil sim weight")
        # if evil_sim_weight > 0.1:
        #     print(evil_sim_weight)
        
        if evil_sim_weight == 0:
            weighted_param_aggregated = (good_weights_prime[j] + (sim_weight * bad_weights_prime[j])) / (1 + sim_weight)
        else:
            weighted_param_aggregated = (good_weights_prime[j] + (sim_weight * bad_weights_prime[j]) + (evil_sim_weight * evil_weights_prime[j])) / (1 + sim_weight + evil_sim_weight)
        aggregate_for_good.append(weighted_param_aggregated)
    
    if name == "conv":
        return good_weights_prime, good_weights_prime
    else:
        return aggregate_for_good, good_weights_prime
    # return good_weights_prime, good_weights_prime  # All Benign

def generateparams(weights, biases, first_hidden_w, first_hidden_b, second_hidden_w, second_hidden_b, conv1_w, conv1_b, conv2_w, conv2_b, comb_C, benign):
    if benign:
        weights_param = np.array(weights)[comb_C == 0]
        biases_param = np.array(biases)[comb_C == 0]

        fh_w = np.array(first_hidden_w)[comb_C == 0]
        fh_b = np.array(first_hidden_b)[comb_C == 0]

        sh_w = np.array(second_hidden_w)[comb_C == 0]
        sh_b = np.array(second_hidden_b)[comb_C == 0]

        c1_w = np.array(conv1_w)[comb_C == 0]
        c1_b = np.array(conv1_b)[comb_C == 0]

        c2_w = np.array(conv2_w)[comb_C == 0]
        c2_b = np.array(conv2_b)[comb_C == 0]

    elif not benign:
        weights_param = np.array(weights)[comb_C == 2]
        biases_param = np.array(biases)[comb_C == 2]

        fh_w = np.array(first_hidden_w)[comb_C == 2]
        fh_b = np.array(first_hidden_b)[comb_C == 2]

        sh_w = np.array(second_hidden_w)[comb_C == 2]
        sh_b = np.array(second_hidden_b)[comb_C == 2]

        c1_w = np.array(conv1_w)[comb_C == 2]
        c1_b = np.array(conv1_b)[comb_C == 2]

        c2_w = np.array(conv2_w)[comb_C == 2]
        c2_b = np.array(conv2_b)[comb_C == 2]

    return weights_param, biases_param, fh_w, fh_b, sh_w, sh_b, c1_w, c1_b, c2_w, c2_b
    
def generateparams_nocluster(weights, biases, first_hidden_w, first_hidden_b, second_hidden_w, second_hidden_b, conv1_w, conv1_b, conv2_w, conv2_b, malicious, benign):
    if not benign:
        bad_index = [index for index,value in enumerate(malicious) if value == "2"]
        weights_param = np.array(weights)[bad_index]
        biases_param = np.array(biases)[bad_index]

        fh_w = np.array(first_hidden_w)[bad_index]
        fh_b = np.array(first_hidden_b)[bad_index]

        sh_w = np.array(second_hidden_w)[bad_index]
        sh_b = np.array(second_hidden_b)[bad_index]

        c1_w = np.array(conv1_w)[bad_index]
        c1_b = np.array(conv1_b)[bad_index]

        c2_w = np.array(conv2_w)[bad_index]
        c2_b = np.array(conv2_b)[bad_index]

    elif benign:
        good_index = [index for index,value in enumerate(malicious) if value == "0"]
        weights_param = np.array(weights)[good_index]
        biases_param = np.array(biases)[good_index]

        fh_w = np.array(first_hidden_w)[good_index]
        fh_b = np.array(first_hidden_b)[good_index]

        sh_w = np.array(second_hidden_w)[good_index]
        sh_b = np.array(second_hidden_b)[good_index]

        c1_w = np.array(conv1_w)[good_index]
        c1_b = np.array(conv1_b)[good_index]

        c2_w = np.array(conv2_w)[good_index]
        c2_b = np.array(conv2_b)[good_index]

    return weights_param, biases_param, fh_w, fh_b, sh_w, sh_b, c1_w, c1_b, c2_w, c2_b

def full_clustering(parameter, client_id, malicious, layer, name, server_round, individual_acc, e, highest_accuracy, lowest_accuracy):
    comb_C, accuracies, benign_class, e = nd_clustering(parameter, client_id, malicious, layer, name, server_round, individual_acc, e)
    
    print("accuracies is", accuracies)

    """Determine benign and malicious clients"""
    highest_acc = 0
    lowest_acc = 10
    record = 0
    acc_diff = 0

    if len(accuracies) > 2:  # if there are more than two clusters, merge clusters so that there is only two clusters
        # print("More than 2 clusters")
        # Find the two distinct clusters by their accuracies
        max_tuple = max(accuracies, key=lambda x:x[0])
        min_tuple = min(accuracies, key=lambda x:x[0])
        # print("max_tuple:", max_tuple, ", min_tuple:", min_tuple)
        remaining = [t for t in accuracies if t != min_tuple and t!= max_tuple]
        while len(remaining) != 0:
            rep = remaining[0]
            # print("rep:", rep)
            if abs(max_tuple[0] - rep[0]) > abs(rep[0] - min_tuple[0]):  #if the accuracy is closer to the min
                # print("closer to min")
                mod_combC = [min_tuple[1] if item == rep[1] else item for item in comb_C]
            else:
                # print("closer to max")
                mod_combC = [max_tuple[1] if item == rep[1] else item for item in comb_C]
            remaining.remove(rep)
            comb_C = mod_combC
            accuracies.remove(rep)
    #     accuracies = [tup for tup in accuracies if tup[0] != -1]   # remove noise cluster
    print("accuracies is", accuracies)

    # if not all(abs(x[0] - accuracies[0][0]) < 0.05 for x in accuracies):  # if the values of the array is very different
    if len(accuracies) == 2:
        if accuracies[0][0] == accuracies[1][0]:
            #larger cluster is benign
            unique_labels = np.unique(comb_C)
            malicious_class = unique_labels[unique_labels != benign_class][0]
            highest_accuracy = accuracies[0][0]
            lowest_accuracy = 0
        else:
            if not all(abs(x[0] - accuracies[0][0]) < 0.05 for x in accuracies):
                record = 1
        
            for acc, cluster in accuracies:
                if (acc > highest_acc):   # if (acc > highest_acc) and (record == 1):
                    benign_class = cluster
                    highest_acc = acc
                if (acc < lowest_acc):
                    malicious_class = cluster
                    lowest_acc = acc
                # print("benign_class, high_acc:", benign_class, highest_acc)
                # print("mali_class, low_acc:", malicious_class, lowest_acc)
                highest_accuracy = highest_acc
                lowest_accuracy = lowest_acc
    else:
        if len(accuracies) == 1:  #if towards the end there is no malicious client, only benign clients. 
            # print("highest accuracy is", highest_accuracy)
            # print("accuracy is", accuracies[0][0])
            if accuracies[0][0] >= highest_accuracy:
                benign_class = accuracies[0][1]
                malicious_class = 10
            elif abs(highest_accuracy - accuracies[0][0]) <= 0.1:
                benign_class = accuracies[0][1]
                malicious_class = 10
            else:
                acc_diff = abs(highest_accuracy - accuracies[0][0])
                benign_class = 10
                malicious_class = accuracies[0][1]
        else:
            unique_labels = np.unique(comb_C)
            malicious_class = unique_labels[unique_labels != benign_class][0]

    print("benign class is", benign_class)
    print("malicious class is", malicious_class)

    # Update comb_C so that benign is 0 and malicious is 1
    mod_combC = [0 if item == benign_class else 2 for item in comb_C]
    comb_C = np.array(mod_combC)
    benign_class = 0

    correct = 0
    for i in range(len(comb_C)):
        if (comb_C[i] == benign_class) and (malicious[i] == "0"):
            correct += 1
        elif (comb_C[i] != benign_class) and (malicious[i] == "2"):
            correct += 1
    clustering_acc = correct / len(malicious)

    print("Clustering acc for", name, "is", clustering_acc)

    # if name == "biases":
    #     ws["U"+str(server_round+4)] = clustering_acc
    # elif name == "conv1w":
    #     ws["V"+str(server_round+4)] = clustering_acc
    # elif name == "conv2w":
    #     ws["W"+str(server_round+4)] = clustering_acc
    # elif name == "fhw":
    #     ws["X"+str(server_round+4)] = clustering_acc
    # elif name == "shw":
    #     ws["Y"+str(server_round+4)] = clustering_acc

    return comb_C, record, acc_diff, highest_accuracy, lowest_accuracy, e