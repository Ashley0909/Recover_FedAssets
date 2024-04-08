"""Print model's state_dict"""
# print("FL Model's state_dict:")
# for param_tensor in self.model.state_dict():
#     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

# # Print optimizer's state_dict
# print("NN Optimizer's state_dict:")
# for var_name in optim.state_dict():
#     print(var_name, "\t", optim.state_dict()[var_name])

"""train the model"""
# def train(net, trainloader, optimizer, epochs, device: str):
#     # Train the network on the training set.

#     # This is a fairly simple training loop for PyTorch.
    
#     criterion = torch.nn.CrossEntropyLoss()
#     net.train()
#     net.to(device)
    
#     # print("len of trainloader is", len(trainloader))
#     #In FL, there are 450 batches each time, and then len(trainloader[0]) is undefined cuz dataloader is not subscriptable
#     #In NN, there are 2 batches for 4 train data, and len(trainloader[0]) is undefined cuz dataloader is not subscriptable

#     # i = 0
#     for j in range(epochs):
#         for images, labels in trainloader:  # in each for loop, there are two images we input
#             # if i == 0 and j == 0:
#                 # print("length of the images is", len(images)) # FL: 20 images  NN: 2 images  (the batch size)
#                 # print("length of images[0]", len(images[0]))  # Fl: 1 tensor object  NN: 30 samples
#                 # print("length of images[0][0]", len(images[0][0])) # 28 samples   NN: 5 samples
#                 # print("length of images[0][0][0] is", len(images[0][0][0])) # 28 samples  NN: error
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             # i += 1


"""Train Neural Network"""
# def nn_train(net, trainloader, optimizer, epochs, device: str):
#     # Train the network on the training set.
#     criterion = torch.nn.CrossEntropyLoss()
#     net.train()
#     net.to(device)

#     for _ in range(epochs):
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#         wandb.log({"train_loss": loss})  #last batch

"""Test neural network"""
# def nn_test(net, testloader, device: str):
#     # Validate the network on the entire test set, and report loss and accuracy.
#     criterion = torch.nn.CrossEntropyLoss()

#     correct = 0
#     net.eval()
#     net.to(device)
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = net(images)
#             loss = criterion(net(images), labels)
#             for result in outputs:
#                 max_arg = torch.argmax(result).int()
#                 result.zero_()
#                 result[max_arg] = 1.0
#             correct += ((outputs == labels).sum().item()//2)
#             wandb.log({"test_loss": loss})

#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy


"""Combine the first and last layer to be a 2D data for clustering"""
        # comb_data = list(zip(layer1_dp, layern_dp))
        # comb_kmeans = KMeans(n_clusters=2, init='k-means++').fit(comb_data)
        # plt.scatter(layer1_dp, layern_dp, c=comb_kmeans.labels_)
        # comb_C = comb_kmeans.predict(comb_data)
        # comb0 = np.array(comb_data)[comb_C == 0]
        # comb1 = np.array(comb_data)[comb_C == 1]
        # clabel0 = np.array(label)[comb_C == 0]
        # clabel1 = np.array(label)[comb_C == 1]
        # texts = []
        # num = 1
        # for i, txt in enumerate(clabel0):
        #     texts.append(plt.text(comb0[i][0], comb0[i][1], txt))
        #     num *= -1
        # for i, txt in enumerate(clabel1):
        #     texts.append(plt.text(comb1[i][0], comb1[i][1], txt))
        #     num *= -1
        # plt.show(block=True)


"""Take the last hidden layer's weight and biases for each neuron"""
# layern_dp = []
# for j in range(len(parameter)):
#     weights = parameter[j][6]
#     biases = parameter[j][7]
#     sum_w = np.sum(weights)
#     sum_b = np.sum(biases)
#     layern_dp.append(sum_w+sum_b)

# # Reshape the two sets of data points
# dp1_reshaped = np.array(layer1_dp).reshape(-1,1)
# dpn_reshaped = np.array(layern_dp).reshape(-1,1)
# conv_reshaped = np.array(conv_dp).reshape(-1,1)

# """Cluster the first layer"""
# kmeans1 = KMeans(n_clusters=2, init='k-means++').fit(dp1_reshaped)
# layer1_C = kmeans1.predict(dp1_reshaped)

# cluster0 = np.array(layer1_dp)[layer1_C == 0]
# cluster1 = np.array(layer1_dp)[layer1_C == 1]
# label0 = np.array(label)[layer1_C == 0]
# label1 = np.array(label)[layer1_C == 1]

# """Cluster the last layer"""
# kmeansn = KMeans(n_clusters=2, init='k-means++').fit(dpn_reshaped)
# layern_C = kmeansn.predict(dpn_reshaped)

# ncluster0 = np.array(layern_dp)[layern_C == 0]
# ncluster1 = np.array(layern_dp)[layern_C == 1]
# nlabel0 = np.array(label)[layern_C == 0]
# nlabel1 = np.array(label)[layern_C == 1]

# """Cluster the conv layer"""
# kmeanconv = KMeans(n_clusters=2, init='k-means++').fit(conv_reshaped)
# conv_C = kmeansn.predict(conv_reshaped)

# conv_cluster0 = np.array(conv_dp)[conv_C == 0]
# conv_cluster1 = np.array(conv_dp)[conv_C == 1]
# conv_label0 = np.array(label)[conv_C == 0]
# conv_label1 = np.array(label)[conv_C == 1]

"""Plotting the scatter plot"""
# _, axs = plt.subplots()
# axs.scatter(cluster0, np.zeros(len(cluster0)), color='tab:blue')
# axs.scatter(cluster1, np.zeros(len(cluster1)), color='tab:orange')
# axs.scatter(ncluster0, np.ones(len(ncluster0)), color='tab:olive')
# axs.scatter(ncluster1, np.ones(len(ncluster1)), color='tab:purple')
# axs.scatter(conv_cluster0, np.ones(len(conv_cluster0)), color='tab:red')
# axs.scatter(conv_cluster1, np.ones(len(conv_cluster1)), color='tab:brown')

# texts = []
# num = 1
# for i, txt in enumerate(label0):
#     texts.append(plt.text(cluster0[i], 0.007*num, txt))
#     num *= -1
# for i, txt in enumerate(label1):
#     texts.append(plt.text(cluster1[i], 0.007*num, txt))
#     num *= -1
# num = 0.02
# for i, txt in enumerate(nlabel0):
#     texts.append(plt.text(ncluster0[i], 1+num, txt))
#     num *= -1
# num = 0.02
# for i, txt in enumerate(nlabel1):
#     texts.append(plt.text(ncluster1[i], 1-num, txt))
#     num *= -1
# num = 0.02
# for i, txt in enumerate(conv_label0):
#     texts.append(plt.text(conv_cluster0[i], 1+num, txt))
#     num *= -1
# num = 0.02
# for i, txt in enumerate(conv_label1):
#     texts.append(plt.text(conv_cluster1[i], 1-num, txt))
#     num *= -1

# """Plotting the explained variance against lambda"""
# cumulative_variance = np.sum(pca.explained_variance_ratio_)

# variance_list = []
# for i in range(1,11):
#     cum_sum = nd_clustering(parameter, malicious, lamb=1)
#     variance_list.append(cum_sum)

# plt.plot(range(1,11), variance_list, 'o-')
# plt.title('PCA Variance Explained vs Lambda')
# plt.xlabel('Lambda')
# plt.ylabel('Cumulative Variance Explained')
# plt.axhline(y=max(variance_list), color='r', linestyle='--')
# plt.show(block=True)

"""Try with KMeans"""
    # kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4)
    # kmeans.fit(reduced_data)
    
    # h = 0.02

    # x_min, x_max = reduced_data[:,0].min() - 0.5, reduced_data[:,0].max() + 0.5
    # y_min, y_max = reduced_data[:,1].min() - 0.5, reduced_data[:,1].max() + 0.5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(
    #     Z,
    #     interpolation="nearest",
    #     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #     cmap=plt.cm.Paired,
    #     aspect="auto",
    #     origin="lower",
    # )

    # plt.plot(reduced_data[:,0], reduced_data[:,1], "k.", markersize=2)
    # plt.title("Clustering of {0} clients".format(len(parameter)))
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    # centroids = kmeans.cluster_centers_
    # comb_C = kmeans.predict(reduced_data)

"""Identify the clusters with smaller number of elements and pass them for statistical check"""
# # Remove the cluster that has the most elements (this will be our benign clients by assumption)
# length_list = [len(comb0), len(comb1)]
# longest = max(length_list)
# benign_class = length_list.index(longest) # 0 or 2

# bad_clients = cid[comb_C != benign_class]
# good_clients = cid[comb_C == benign_class]  
# print("bad clients:", bad_clients)
# print("good clients:", good_clients)


"""Approach 1: Take mean and variance of the good clients for each filter  (slower)""" 
# count = [0]*len(good_biases[0])
# for i in range(len(good_biases[0])):  # number of filters
#         mean = np.mean(np.array(good_biases[:,i]))
#         std = np.std(np.array(good_biases[:,i]))

#         for j in range(len(bad_biases)):  # for each suspicious client
#         p = bad_biases[:,i][j]
#         z_score = (p - mean) / std
#         if abs(z_score) >= 2.0:
#                 count[i] += 1

# target_label = np.argmax(np.array(count))


"""Cross Check two approaches to make sure the attackers are using single label attack"""
# if target_label == target_label2:
#     print("Single Target Attack Detected from Malicious Clients, the target label is", target_label)
# else:
#     print("Target_label by 1:", target_label, "Target Label by 2:", target_label2)
#     print("Multiple Target Labels detected")


"""Aggregate Convolutional Layer"""
# def aggregate_conv(good_c1_w, good_c1_b, bad_c1_w, bad_c1_b, good_c2_w, good_c2_b, bad_c2_w, bad_c2_b, evil_conv1_w, evil_conv1_b, evil_conv2_w, evil_conv2_b, evil_parameter, evil_numexamples, good_num_examples, parameter, good_clients, bad_num_examples, bad_clients, acc_diff):
#     """Aggregate all good client's original expanded weights"""
#     print("Aggregate conv")
#     conv_weights = []
#     num_examples_total = sum(good_num_examples)
#     bad_num_total = sum(bad_num_examples) 
#     evil_num_total = sum(evil_numexamples)

#     good_c1b_aggre = naive_aggregate(good_c1_b, good_num_examples)
#     bad_c1b_aggre = naive_aggregate(bad_c1_b, bad_num_examples)
#     good_c2b_aggre = naive_aggregate(good_c2_b, good_num_examples)
#     bad_c2b_aggre = naive_aggregate(bad_c2_b, bad_num_examples)

#     good_c1w_aggre = naive_aggregate(good_c1_w, good_num_examples)
#     bad_c1w_aggre = naive_aggregate(bad_c1_w, bad_num_examples)
#     good_c2w_aggre = naive_aggregate(good_c2_w, good_num_examples)
#     bad_c2w_aggre = naive_aggregate(bad_c2_w, bad_num_examples)      

#     for k in range(4): # 4 convolutional layers
#         weighted_weights = []
#         for j, i in enumerate(good_clients): #each benign client
#             weighted_weights.append(parameter[int(i)][k] * good_num_examples[j])
#         weights_prime: NDArrays = [
#             reduce(np.add, layer_updates) / num_examples_total
#             for layer_updates in zip(*weighted_weights)
#         ]

#         bad_weighted = []
#         for j, i in enumerate(bad_clients):
#             bad_weighted.append(parameter[int(i)][k] * bad_num_examples[j])
#         bad_weights_prime: NDArrays = [
#             reduce(np.add, layer_updates) / bad_num_total
#             for layer_updates in zip(*bad_weighted)
#         ]  

#         evil_weighted = []
#         if evil_num_total != 0:
#             for i in range(len(evil_parameter)):
#                 evil_weighted.append(evil_parameter[i][k] * evil_numexamples[i])
#             evil_weights_prime: NDArrays = [
#                 reduce(np.add, layer_updates) / evil_num_total
#                 for layer_updates in zip(*evil_weighted)
#             ]
#             if k == 0:
#                 evil_aggregated = naive_aggregate(evil_conv1_w, evil_numexamples)
#             elif k == 1:
#                 evil_aggregated = naive_aggregate(evil_conv1_b, evil_numexamples)
#             elif k == 2:
#                 evil_aggregated = naive_aggregate(evil_conv2_w, evil_numexamples)
#             elif k == 3:
#                 evil_aggregated = naive_aggregate(evil_conv2_b, evil_numexamples)
#         else:
#             evil_aggregated = []


#         if bad_num_total == 0:                      # If no bad clients, only aggregate good clients
#             conv_weights.append(weights_prime)
#             continue
#         elif num_examples_total == 0:               # If no good clients, aggregate bad clients with acc_diff
#             plate_array = []
#             for j in range(len(parameter[0][k])): # per neuron
#                 weighted_param_aggregated = np.exp(-9 * acc_diff) * np.array(bad_weights_prime[j])
#                 plate_array.append(weighted_param_aggregated.tolist())
#             conv_weights.append(plate_array)
#             continue

#         plate_array = []

#         if k == 0:
#             for j in range(len(parameter[0][0])):
#                 if evil_aggregated != []:
#                     evil_dist = abs(good_c1w_aggre[j] - evil_aggregated[j])
#                     evil_similarity = np.exp(-12 * evil_dist)
#                 else:
#                     evil_similarity = 0
#                 dist = abs(good_c1w_aggre[j] - bad_c1w_aggre[j])
#                 similarity = np.exp(-9 * dist)
#                 # similarity = 0
#                 if evil_similarity == 0:
#                     weighted_param_aggregated = (((similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity)).tolist()
#                 else:
#                     weighted_param_aggregated = (((evil_similarity * evil_weights_prime[j]) + (similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity + evil_similarity)).tolist()
#                 plate_array.append(weighted_param_aggregated)
#         elif k == 1:
#             for j in range(len(good_c1_b[0])):
#                 if evil_aggregated != []:
#                     evil_dist = abs(good_c1b_aggre[j] - evil_aggregated[j])
#                     evil_similarity = np.exp(-12 * evil_dist)
#                 else:
#                     evil_similarity = 0
#                 dist = abs(good_c1b_aggre[j] - bad_c1b_aggre[j])
#                 similarity = np.exp(-9 * dist)
#                 # similarity = 0
#                 if evil_similarity == 0:
#                     weighted_param_aggregated = (((similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity)).tolist()
#                 else:
#                     weighted_param_aggregated = (((evil_similarity * evil_weights_prime[j]) + (similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity + evil_similarity)).tolist()
#                 plate_array.append(weighted_param_aggregated)
#         elif k == 2:
#             for j in range(len(parameter[0][2])):
#                 if evil_aggregated != []:
#                     evil_dist = abs(good_c2w_aggre[j] - evil_aggregated[j])
#                     evil_similarity = np.exp(-12 * evil_dist)
#                 else:
#                     evil_similarity = 0
#                 dist = abs(good_c2w_aggre[j] - bad_c2w_aggre[j])
#                 similarity = np.exp(-9 * dist)
#                 # similarity = 0
#                 if evil_similarity == 0:
#                     weighted_param_aggregated = (((similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity)).tolist()
#                 else:
#                     weighted_param_aggregated = (((evil_similarity * evil_weights_prime[j]) + (similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity + evil_similarity)).tolist()
#                 plate_array.append(weighted_param_aggregated)
#         elif k == 3:
#             for j in range(len(good_c2_b[0])):
#                 if evil_aggregated != []:
#                     evil_dist = abs(good_c2b_aggre[j] - evil_aggregated[j])
#                     evil_similarity = np.exp(-12 * evil_dist)
#                 else:
#                     evil_similarity = 0
#                 dist = abs(good_c2b_aggre[j] - bad_c2b_aggre[j])
#                 similarity = np.exp(-10 * dist)
#                 # similarity = 0
#                 if evil_similarity == 0:
#                     weighted_param_aggregated = (((similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity)).tolist()
#                 else:
#                     weighted_param_aggregated = (((evil_similarity * evil_weights_prime[j]) + (similarity * bad_weights_prime[j]) + weights_prime[j]) / (1 + similarity + evil_similarity)).tolist()
#                 plate_array.append(weighted_param_aggregated)
#         conv_weights.append(plate_array)

#         # final_param_aggregated = ((np.array(weights_prime) + np.array(bad_weights_prime)) / 2).tolist() #Aggregate Equally
#         # conv_weights.append(final_param_aggregated) #Aggregate Equally
    
#     fconv_w = conv_weights[0]
#     fconv_b = conv_weights[1] 
#     lconv_w = conv_weights[2]
#     lconv_b = conv_weights[3]

#     return fconv_w, fconv_b, lconv_w, lconv_b

"""Original Convert from weights to results"""
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]

        # num_examples = [fit_res.num_examples for _, fit_res in results]  #original
        # all_id = [(i, cp.cid) for i, (cp, _) in enumerate(results)]   #original
        # malicious = [str(fit_res.metrics["malicious"]) for _, fit_res in results]     #original

"""Decision of record in around line 498"""
# if abs(acc - highest_acc) > 0.1:  # if the two rep accuracies have more than 10% difference, then we record
    #     record = 1
    # else:
    #     record = 0

"""Finding clients that is closest to the centroid"""
# closest_points_idx, _ = pairwise_distances_argmin_min(centroids, reduced_data)   #find the closest point locally to the centroids
# closest_client = cid[closest_points_idx]   #corresponding clients globally

"""Computing accuracies of representatives"""
# for x in selected_clients:
#     _, fit_res = new_results[x]
#     parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
#     eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
#     if eval_res is None:
#         return None
#     _, metrics = eval_res
#     # print("malicious?", int(malicious[x]), "cluster assignment:", comb_C[x], "accuracy:", metrics["accuracy"])
#     if comb_C[x] not in unique_dict:
#         unique_dict[comb_C[x]] = (comb_C[x], metrics["accuracy"])
#     elif unique_dict[comb_C[x]][1] < metrics["accuracy"]:
#         unique_dict[comb_C[x]] = (comb_C[x], metrics["accuracy"])

# accuracies = list(unique_dict.values())
# print("unique_list", accuracies)

"""Generate a dictionary of determined benign and malicious clients"""
# determined_status = {}
# for i in range(len(weights_results)):
#     if local_cid[i] in good_clients:
#         determined_status[client_id[i]] = 0
#     elif local_cid[i] in bad_clients:
#         determined_status[client_id[i]] = 2

"""Plotting these graphs if the clustering is correct"""
# show_clustered_plots(good_clients, bad_clients, 
#        good_fh_w, bad_fh_w, good_fh_b, bad_fh_b,
#        good_sh_w, bad_sh_w, good_sh_b, bad_sh_b,
#        good_weights, bad_weights, good_biases, bad_biases)

"""Plotting these graphs if the clustering is wrong"""
# show_plots(client_id, malicious,
#        first_hidden_w, first_hidden_b,
#        second_hidden_w, second_hidden_b,
#        weights, biases)

"""Aggregating convolutional layers"""
# fconv_w, fconv_b, lconv_w, lconv_b = aggregate_conv(good_conv1_w, good_conv1_b, bad_conv1_w, bad_conv1_b, good_conv2_w, good_conv2_b, bad_conv2_w, bad_conv2_b, evil_conv1_w, evil_conv1_b, evil_conv2_w, evil_conv2_b, evil_parameter, evil_numexamples,
#                                                      good_num_examples, parameter, good_clients, bad_num_examples, bad_clients, acc_diff)

# final_aggregated.append(fconv_w)
# final_aggregated.append(fconv_b)
# final_aggregated.append(lconv_w)
# final_aggregated.append(lconv_b)

"""Aggregate the parameters"""
# parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

"""Print clustering accuracy for each layer"""
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



#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this