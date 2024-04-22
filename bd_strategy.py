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
wb = load_workbook( "Results.xlsx" )
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
benign_record = []
final_model = []
final_metric = []
e = 0
flag = 0
benign_average, malicious_average, global_targetlabel = None, None, None

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

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays)
        # eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {}, torch.device("cuda"))   #GPU
        if eval_res is None:
            return None
        loss, metrics = eval_res

        # Computing the poisoning accuracy
        attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays)
        # attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays, {}, torch.device("cuda")) #GPU
        if attack_eval_res is None:
            return None
        _, attack_metrics = attack_eval_res

        if server_round > 0:
            print("Global Poisoning Accuracy:", attack_metrics["accuracy"])
            ws[constant.EXCEL_CELL+str(server_round+315)] = attack_metrics["accuracy"]
        
        """Save Results"""
        wb.save( "Results.xlsx" )
            
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
        global malicious_record, final_model, final_metric, global_targetlabel, e, benign_record, benign_average, malicious_average, flag

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

        evil_numexamples = []
        evil_parameter = []
        evil_results = []
        i = 0
        total = 0
        count = 0
        for cp, fit_res in results:
            if cp.cid in malicious_record:
                evil_numexamples.append(fit_res.num_examples)
                evil_parameter.append(parameters_to_ndarrays(fit_res.parameters))
                evil_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                """Computing individual poisoning accuracies"""
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays)
                # attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays, {}, torch.device("cuda"))  #GPU
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
                i += 1

        print("malicious is", malicious)
        
        num_examples = [res[1] for res in weights_results]
        print("Number of Evil Clients:", len(evil_results))

        # if len(num_examples) == 0:
        #     print("Only Evil Clients in this round, void this round.")
        #     return final_model, final_metric
        
        client_id = np.array(all_id)[:,1]
        local_cid = np.array(all_id)[:,0]
        parameter = [client[0] for client in weights_results]  #[c1:[10array], c2:[10array], ...]  #client[1] is the number of examples

        print("Number of Preset Malicious Clients is", malicious.count("2"))
        print("Number of Preset Benign Clients is", malicious.count("0"))

        """Check backdoor task accuracy of this round's attackers"""
        for x in range(len(new_results)):
            if malicious[x] == '2':
                _, fit_res = new_results[x]
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                attack_eval_res = self.attack_evaluate_fn(server_round, parameters_ndarrays)
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

        ws[constant.EXCEL_CELL+str(server_round+211)] = poisoning_acc

        """Get FC Weight for clustering"""
        fcw = []
        for i in range(len(parameter)): 
            w = 0
            vector = []  #set up a vector for each client
            for j in range(len(parameter[i][-2])): #10
                w = np.sum(parameter[i][-2][j]) #84
                vector.append(w)
            fcw.append(np.array(vector))

        if len(evil_results) > 0:
            evil_fcw = []
            for i in range(len(evil_results)): 
                w = 0
                vector = []  #set up a vector for each client
                for j in range(len(evil_parameter[i][-2])): #10
                    w = np.sum(evil_parameter[i][-2][j]) #84
                    vector.append(w)
                evil_fcw.append(np.array(vector))
        else:
            evil_fcw = []

        comb_C, e, flag = nd_clustering(parameter, client_id, malicious, fcw, "fcw", server_round, e, flag)

        if benign_average == None and malicious_average == None:  # KMeans and 2 clusters
            """Allocate good and bad clients"""
            bad_clients = local_cid[comb_C == 2]
            good_clients = local_cid[comb_C == 0]

            """Split the parameters into good and malicious"""
            good_fcw = np.array(fcw)[comb_C == 0]
            bad_fcw = np.array(fcw)[comb_C == 2]

            print("Originally, length of good results is", len(good_clients), "and length of bad results is", len(bad_clients))
            print("Originally, comb_C is", comb_C)

            """Assume Clustering 100%"""
            # bad_index = [index for index,value in enumerate(malicious) if value == "2"]
            # good_index = [index for index,value in enumerate(malicious) if value == "0"]
            # bad_clients = local_cid[bad_index]
            # good_clients = local_cid[good_index]

            # good_fcw = np.array(fcw)[good_index]
            # bad_fcw = np.array(fcw)[bad_index]

            """Detecting Target Label in the first round"""
            if (len(good_clients) > 0) and (len(bad_clients) > 0 or len(evil_results) > 0):
                dist_list = []
                sign_list = []
                bad_averages = []
                good_averages = []
                for i in range(len(good_fcw[0])):
                    good_biases_average = compute_average(good_fcw[:,i], len(good_clients))
                    if len(bad_clients) == 0:
                        bad_biases_average = compute_average(np.array(evil_fcw)[:,i], len(evil_results))
                    elif len(evil_results) == 0:
                        bad_biases_average = compute_average(bad_fcw[:,i], len(bad_clients))
                    else:
                        bad_biases_average = compute_average(np.concatenate((bad_fcw, evil_fcw), axis=0)[:,i], (len(bad_clients)+len(evil_results)))

                    dist = abs(good_biases_average - bad_biases_average)
                    sign = np.sign(good_biases_average - bad_biases_average)
                    sign_list.append(sign)
                    dist_list.append(dist)
                    bad_averages.append(bad_biases_average)
                    good_averages.append(good_biases_average)

                target_label = np.argmax(np.array(dist_list))
                if sign_list[target_label] == 1:
                    print("good > bad, ALERT!!")
                    comb_C = np.array([2 if x == 0 else 0 if x == 2 else x for x in comb_C])
                    benign_average = bad_averages[target_label]
                    malicious_average = good_averages[target_label]
                else:
                    print("bad > good, ok!")
                    benign_average = good_averages[target_label]
                    malicious_average = bad_averages[target_label]
                global_targetlabel = target_label
                record = 1
                acc_diff = 0
        else:
            comb_C, record, acc_diff = merge_clients(comb_C, fcw, local_cid, benign_average, malicious_average, global_targetlabel)

        print("Target label is", global_targetlabel)
        print("Now, comb_C is", comb_C)

        heatmaps(local_cid, comb_C, evil_fcw, np.array(fcw), 'FCW', server_round)

        if record == 1:
            global_bad = client_id[comb_C == 2]
            # global_bad = client_id[bad_index]
            print(len(global_bad), "added to evil list")
            malicious_record.extend(global_bad)
            malicious_record = list(set(malicious_record)) #avoid duplicates

            global_good = client_id[comb_C == 0]
            benign_record.extend(global_good)
            benign_record = list(set(benign_record))

        """Compute accuracies"""
        correct = 0
        for i in range(len(comb_C)):
            if (comb_C[i] == 0) and (malicious[i] == "0"):
                correct += 1
            elif (comb_C[i] == 2) and (malicious[i] == "2"):
                correct += 1
        clustering_acc = correct / len(malicious)

        print("Final Clustering acc is", clustering_acc)
        
        ws[constant.EXCEL_CELL+str(server_round+107)] = clustering_acc

        """After detecting the clients and their target label, make a function that determines the weight of contribution"""
        good_results = [weights_results[i] for i in range(len(weights_results)) if comb_C[i] == 0]
        bad_results = [weights_results[i] for i in range(len(weights_results)) if comb_C[i] == 2]

        """Assume Clustering 100%"""
        # good_results = [weights_results[i] for i in range(len(weights_results)) if i in good_index]
        # bad_results = [weights_results[i] for i in range(len(weights_results)) if i in bad_index]

        print("length of good results is", len(good_results), "and length of bad results is", len(bad_results))

        parameters_aggregated = ndarrays_to_parameters(resnet_aggregate(good_results, bad_results, evil_results, acc_diff, global_targetlabel))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Record the final model in case if the next round is a void round
        # final_model = parameters_aggregated
        # final_metric = metrics_aggregated

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
        valid_results = []
        print("benign record is", benign_record)
        if benign_record != []:
            for cp, evaluate_res in results:
                if cp.cid in benign_record:
                    valid_results.append((evaluate_res.num_examples, evaluate_res.loss))

            loss_aggregated = weighted_loss_avg(valid_results)
        else:
            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in results
                ]
            )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            if benign_record != []:
                eval_metrics = []
                for cp, res in results:
                    if cp.cid in benign_record:
                        eval_metrics.append((res.num_examples, res.metrics))
            else:
                eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        if server_round > 0:
            print("Federated Accuracy:", metrics_aggregated["accuracy"])
            ws[constant.EXCEL_CELL+str(server_round+4)] = metrics_aggregated["accuracy"]
        
        """Save Results"""
        wb.save( "Results.xlsx" )

        return loss_aggregated, metrics_aggregated

def nd_clustering(parameter, cid, malicious, layer, name, server_round, e, flag):
    label = []
    textstr = ''
    for k in range(len(parameter)):
        label.append("Client" + str(cid[k]) + "=>" + malicious[k] + "\n")
        textstr += f'Client {str(cid[k])} => {malicious[k]} \n'
    
    if len(parameter) < 2:
        return [-1], -1, e, flag
        
    """Run PCA on the n dimensional data"""
    """2D"""
    pca = PCA(n_components=2)
    """3D"""
    # pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(layer)
    
    if flag == 0:
        print("KMeans")
        kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4).fit(reduced_data)
        comb_C = kmeans.predict(reduced_data)

        centroids = kmeans.cluster_centers_
        centroids_assignment = kmeans.predict(centroids)
        unique_labels = np.unique(comb_C)
        # Compute intra-cluster distances
        intra_cluster = euclidean_distances(centroids[centroids_assignment == unique_labels[0]], centroids[centroids_assignment == unique_labels[1]])
        print("intra_cluster distance is", intra_cluster[0][0])
        if intra_cluster[0][0] < 0.05:
            flag = 1

        max_dist = []
        for l in unique_labels:
            distances = euclidean_distances(reduced_data[comb_C == l], centroids[centroids_assignment == l])
            max_dist.append(np.max(distances))
        e = np.max(max_dist)
    else:
        mp = 5
        e -= 0.0025*(server_round/5)
        mp -= (server_round//20) 
        db = DBSCAN(eps=max(e, 0.03), min_samples=max(3,mp)).fit(reduced_data)
        comb_C = db.labels_

    """Plotting the clusters"""
    """3D"""
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    """2D"""
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
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], edgecolors='k', s=50, label='Cluster {}'.format(l))  # 2D
        # ax.scatter(xy[:, 0], xy[:, 1], xy[:,2], c=[color], edgecolors='k', s=50, label='Cluster {}'.format(l))  #3D

    if flag == 0:
        plt.title("Kmeans Clustering {0} of {1} clients".format(name, len(parameter)))
    else:
        plt.title("DBSCAN Clustering {0} of {1} clients".format(name, len(parameter)))
    
    plt.legend()

    comb0 = np.array(reduced_data)[comb_C == 0]
    comb1 = np.array(reduced_data)[comb_C != 0]
    clabel0 = np.array(label)[comb_C == 0]
    clabel1 = np.array(label)[comb_C != 0]
    texts = []
    num = 1
    for i, txt in enumerate(clabel0):
        # texts.append(ax.text(comb0[i][0], comb0[i][1], comb0[i][2], txt))  #3D
        texts.append(plt.text(comb0[i][0], comb0[i][1], txt))  #2D
        num *= -1
    for i, txt in enumerate(clabel1):
        # texts.append(ax.text(comb1[i][0], comb1[i][1], comb1[i][2], txt))   #3D
        texts.append(plt.text(comb1[i][0], comb1[i][1], txt))   #2D
        num *= -1

    # plt.savefig('clusters/{0}, Round {1}.png'.format(name, server_round))
    """All Benign"""
    # plt.savefig('clusters/AllBenign/{0}, Round {1}.png'.format(name, server_round))

    plt.close()

    """Set the largest cluster to be benign as a default"""
    counts = Counter(comb_C)
    benign_class = max(counts, key=counts.get)

    # Update comb_C so that benign is 0 and malicious is 2 (For only 2 clusters, and only 1 cluster (assume all to be good))
    if len(unique_labels) < 3:
        mod_combC = [0 if item == benign_class else 2 for item in comb_C]
        comb_C = np.array(mod_combC)

    return comb_C, e, flag

def heatmaps(local_cid, comb_C, evil_layer, layer, name, server_round):
    textstr = ''
    good_client = local_cid[comb_C == 0]
    bad_client = local_cid[comb_C == 2]

    for i in range(len(good_client)):
        textstr += f'Client {str(good_client[i])} => 0 \n'
    for i in range(len(bad_client)):
        textstr += f'Client {str(bad_client[i])} => 2 \n'

    good_layer = layer[comb_C == 0]
    bad_layer = layer[comb_C == 2]

    if evil_layer != []:
        plt.imshow(np.concatenate((good_layer, bad_layer, np.array(evil_layer)), axis=0), cmap='viridis', interpolation='nearest')
    else:
        plt.imshow(np.concatenate((good_layer, bad_layer), axis=0), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.annotate(textstr, xy=(0,0.5), verticalalignment='center',  horizontalalignment='left', xycoords='figure fraction')
    plt.xlabel(name)
    plt.ylabel("Clients")
    plt.title("Heatmap of all clients' {0} in Round {1} (Coloured Trigger)".format(name, server_round))
    # if server_round == 1 or server_round % 20 == 0:
    #     plt.savefig('heatmaps/Round {} (Coloured Trigger)'.format(server_round))

    """All Benign"""
    # plt.savefig('heatmaps/AllBenign/{0} in Round {1}.png'.format(name, server_round))

    plt.close()
mo
def compute_average(data, count):
    average = np.sum(data, axis=0) / count

    return average

def resnet_aggregate(good_result, bad_result, evil_result, acc_diff, target_label):
    print("acc_diff is", acc_diff)
    good_numex_total = sum([num_examples for _, num_examples in good_result])
    bad_numex_total = sum([num_examples for _, num_examples in bad_result])
    evil_numex_total = sum([num_examples for _, num_examples in evil_result])

    # Create a list of weights, each multiplied by the related number of examples
    good_weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in good_result]
    
    good_prime: NDArrays = [
        reduce(np.add, layer_updates) / good_numex_total
        for layer_updates in zip(*good_weighted_weights)
    ]

    """All Benign"""
    return good_prime

    bad_weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in bad_result]
    evil_weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in evil_result]

    bad_prime: NDArrays = [
        reduce(np.add, layer_updates) / bad_numex_total
        for layer_updates in zip(*bad_weighted_weights)
    ]

    if evil_numex_total != 0:
        evil_prime: NDArrays = [
            reduce(np.add, layer_updates) / evil_numex_total
            for layer_updates in zip(*evil_weighted_weights)
        ]
    else:
        evil_prime = []

    # For each layer, compute the distance between the good and the bad, then apply the similarity weight to the bad clients
    # Depending on the layer being weight or bias, we have different approaches: directly calc dist for bias since one value per neuron, but sum all weights of a neuron before calc dist
    aggregated_result = []
    if acc_diff == 0:
        for l in range(len(good_prime)-2, len(good_prime)):
            if len(good_prime[l].shape) > 1:
                # weights: sum up all incoming weights
                if bad_prime != []:
                    neuron_dists = list(map(abs, map(lambda x,y: x - y, [sum(x) for x in good_prime[l]], [sum(y) for y in bad_prime[l]])))
                    sim_weight_list = [np.exp(constant.MALI_LAMBDA * neuron_dists[i]) if i != target_label else 0 for i in range(len(neuron_dists))]
                if evil_prime != []:
                    evil_dists = list(map(abs, map(lambda x,y: x - y, [sum(x) for x in good_prime[l]], [sum(y) for y in evil_prime[l]])))
                    evil_sim_list = [np.exp(constant.EVIL_LAMBDA * evil_dists[i]) if i != target_label else 0 for i in range(len(evil_dists))]

                if bad_prime != [] and evil_prime != []:
                    weighted_param_aggregated = [
                        [
                            (g+(s*b)+(es*e)) / (1+s+es)
                            for g,b,e in zip(good_sublist, bad_sublist, evil_sublist)
                        ]
                        for good_sublist, bad_sublist, evil_sublist, s, es in zip(good_prime[l], bad_prime[l], evil_prime[l], sim_weight_list, evil_sim_list)
                    ]
                elif bad_prime != []:
                    weighted_param_aggregated = [
                        [
                            (g+(s*b)) / (1+s)
                            for g,b, in zip(good_sublist, bad_sublist)
                        ]
                        for good_sublist, bad_sublist, s in zip(good_prime[l], bad_prime[l], sim_weight_list)
                    ]
                elif evil_prime != []:
                    weighted_param_aggregated = [
                        [
                            (g+(es*e)) / (1+es)
                            for g,e, in zip(good_sublist, evil_sublist)
                        ]
                        for good_sublist, evil_sublist, es in zip(good_prime[l], evil_prime[l], evil_sim_list)
                    ]
                else:
                    weighted_param_aggregated = good_prime[l]

            elif len(good_prime[l].shape) < 1:
                if evil_prime != [] and bad_prime != []:
                    dist = abs(good_prime[l] - bad_prime[l])
                    sim_weight = np.exp(constant.MALI_LAMBDA * dist)
                    evil_dist = abs(good_prime[l] - evil_prime[l])
                    evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
                    weighted_param_aggregated = (good_prime[l] + (sim_weight * bad_prime[l]) + (evil_sim_weight * evil_prime[l]))/(1+sim_weight+evil_sim_weight)
                elif bad_prime != []:
                    dist = abs(good_prime[l] - bad_prime[l])
                    sim_weight = np.exp(constant.MALI_LAMBDA * dist)
                    weighted_param_aggregated = (good_prime[l] + (sim_weight * bad_prime[l]))/(1+sim_weight)
                elif evil_prime != []:
                    evil_dist = abs(good_prime[l] - evil_prime[l])
                    evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
                    weighted_param_aggregated = (good_prime[l] + (evil_sim_weight * evil_prime[l]))/(1+evil_sim_weight)
                else:
                    weighted_param_aggregated = good_prime[l]
            else:
                if evil_prime != [] and bad_prime != []:
                    neuron_dists = list(map(abs, map(lambda x,y: x - y, good_prime[l], bad_prime[l])))
                    sim_weight_list = [np.exp(constant.MALI_LAMBDA * neuron_dists[i]) if i != target_label else 0 for i in range(len(neuron_dists))]
                    evil_dist = abs(good_prime[l] - evil_prime[l])
                    evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
                    weighted_param_aggregated = list(map(lambda g,b,e,s,es: (g + (s * b) + (es * e))/ (1 + s + es), good_prime[l], bad_prime[l], evil_prime[l], sim_weight_list, evil_sim_weight)) 
                elif bad_prime != []:
                    neuron_dists = list(map(abs, map(lambda x,y: x - y, good_prime[l], bad_prime[l])))
                    sim_weight_list = [np.exp(constant.MALI_LAMBDA * neuron_dists[i]) if i != target_label else 0 for i in range(len(neuron_dists))]
                    weighted_param_aggregated = list(map(lambda g,b,s: (g + (s * b))/ (1 + s), good_prime[l], bad_prime[l], sim_weight_list)) 
                elif evil_prime != []:
                    evil_dist = abs(good_prime[l] - evil_prime[l])
                    evil_sim_weight = np.exp(constant.EVIL_LAMBDA * evil_dist)
                    weighted_param_aggregated = list(map(lambda g,e,es: (g + (es * e))/ (1 + es), good_prime[l], evil_prime[l], evil_sim_weight)) 
                else:
                    weighted_param_aggregated = good_prime[l]

            aggregated_result.append(weighted_param_aggregated)  # records each layer

    # Still need to consider the case when there is no good client
    elif bad_prime != [] and evil_prime != []:
        sim_weight = np.exp(constant.MALI_LAMBDA * acc_diff)
        evil_sim_weight = np.exp(constant.EVIL_LAMBDA * acc_diff)
        for l in range(len(bad_prime)):
            weighted_param_aggregated = ((sim_weight * bad_prime[l]) + (evil_sim_weight * evil_prime[l]))/(sim_weight + evil_sim_weight)
            aggregated_result.append(weighted_param_aggregated)  # records each layer

    elif bad_prime != []:
        sim_weight = np.exp(constant.MALI_LAMBDA * acc_diff)
        for l in range(len(bad_prime)):
            weighted_param_aggregated = (sim_weight * bad_prime[l])/sim_weight
            aggregated_result.append(weighted_param_aggregated)  # records each layer

    elif evil_prime != []:
        evil_sim_weight = np.exp(constant.EVIL_LAMBDA * acc_diff)
        for l in range(len(evil_prime)):
            weighted_param_aggregated = (sim_weight * evil_prime[l])/evil_sim_weight
            aggregated_result.append(weighted_param_aggregated)  # records each layer

    good_prime[-2] = aggregated_result[0]
    good_prime[-1] = aggregated_result[1]
    return good_prime

def merge_clients(comb_C, fcw, local_cid, benign_average, malicious_average, global_targetlabel):
    unique_labels = np.unique(comb_C)
    client_status = {elem: 2 for elem in unique_labels}  # set all clusters to be malicious first
    all_malicious = 1
    record = 1
    acc_diff = 0

    for l in unique_labels:
        c_fcw = np.array(fcw)[comb_C == l]
        c_ids = local_cid[comb_C == l]
        c_average = compute_average(c_fcw[:,global_targetlabel], len(c_ids))
        if abs(malicious_average - c_average) > abs(benign_average - c_average):  #the status should be benign
            client_status[l] = 0
            all_malicious = 0

    if len(unique_labels) == 1:
        record = 0

    if all_malicious == 1: # if there is no benign clients, we need to find acc_diff
        acc_diff = abs(malicious_average - c_average)

    mod_combC = [2 if client_status[l] == 2 else 0 for l in comb_C]
    comb_C = np.array(mod_combC)

    return comb_C, record, acc_diff