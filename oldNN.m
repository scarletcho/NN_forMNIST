%% ANN assignment4: Neural Network Building from scratch
% Q1. Applying techniques to prevent overfitting such as using cross-entropy
% and log-likelihood cost functions (instead of the mean-squared error, or MSE)
% and regularization techniques.
% Select the optimally chosen three-layers (input, hidden, and output layers)
% network conducted in (b-ii) of HW#3.
%----------------------------------------------------------------------------
% (a) Using the same network architecture, apply the cross-entropy cost
% function and log-likelihood cost function.
%----------------------------------------------------------------------------
% (i) Derive the learning algorithm when the cost function is the cross-
%     entropy cost function. Implement the algorithm and perform the MNIST
%     classification. Obtain the results and discuss about the results also
%     comparing the results from the MSE cost function used in HW#3.
% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)
%%
clc;clear all;close all;

n_hidden_set = 50;
mini_batch_size = 20;
eta_set = 0.2;
n_epochs = 30;
cost_function = 'cross_entropy'; % or 'MSE';

%% Load data
addpath('./functions')
load mnist_uint8_matlab.mat

rng('default')
train_x = double(train_x)'/255; % normalize by maximum value (=255)
train_y = double(train_y)';
test_x = double(test_x)'/255;   % normalize by maximum value (=255)
test_y = double(test_y)';

% [NOTE] train_x, train_y, test_x, test_y: 
%        re-organized into (NODES) by (SAMPLES)

%% Training
for etanum = 1:numel(eta_set)
for hnum = 1:numel(n_hidden_set)
%% Q1-(b). Creating a 3-layered neural network
% (b) Try creating a network with three layers ? input layer (784 neurons/nodes),
%     hidden, and output (10 neurons/nodes) layers. 
%---------------------------------------------------------------------------
% (i)  Derive the learning algorithms for the weights and biases of the network.
% (ii) Train the network using your own implementation in (a-ii). Try each 
%      of the hidden layers with 15, 30, and 100 hidden neurons/nodes.
%      Choose the mini-batch size optimally chosen in (a-ii). Show the results
%      and discuss about the results such as the learning curves and final 
%      classification accuracies.
%% (1) Initial parameter set-up
n_hidden = n_hidden_set(hnum);
netsize = [size(train_x,1) n_hidden size(train_y,1)];

biases = {randn(netsize(2),1)*.01, randn(netsize(3),1)*.01};  % initial biases
weights = {randn(netsize(1),netsize(2))*.01, randn(netsize(2),netsize(3))*.01}; % initial weights

%% (2) Stochastic Gradient Descent
% The 1st for loop: epochs
for epochs = 1:n_epochs
    eta = eta_set(etanum);
    training_data = shuffle([train_x ; train_y], 'column');
    train_x = training_data(1:size(train_x,1),:);
    train_y = training_data(size(train_x,1)+1:end,:);
    
    [mini_x, mini_y] = batch_division(train_x, train_y, mini_batch_size);
    %  cf. Better if shuffled again!
    n_mini_batch = size(mini_x,2);
    % [NOTE] mini_x & mini_y: 1 by m cell    (cf. m: number of mini-batches)
    %        each cell: (nodes) by (mini_batch_size)
    
    % The 2nd for loop: mini batches
    for mini_batch = 1:n_mini_batch
        mini_x_singlebatch = mini_x{1,mini_batch};
        mini_y_singlebatch = mini_y{1,mini_batch};
        clear a_next z_next delta stack_a stack_z
        
        % The 3rd for loop: layers
        [stack_a, stack_z] = feedforward_mlp(mini_x_singlebatch,weights,biases);

        % output error of the final layer (BP1)
        delta{1,1} = cost_derivative(stack_a{1,end}, mini_y_singlebatch, cost_function)...
                       .* sigmoid_prime(stack_z{1,end});
        
        % backpropagation (BP2) toward the earlier layers
        for layer = size(netsize,2)-1:-1:2
            delta = [delta, weights{layer} * delta{1,1} .* sigmoid_prime(stack_z{1,layer-1})];
        end
        
        delta = fliplr(delta);
        
        % gradient of C w.r.t. biases & weights
        gradient_w{1} = mini_x_singlebatch*delta{1}'/size(delta{1},2);
        gradient_b{1} = mean(delta{1},2)/size(delta{1},2);
        
        for layer = 2:size(netsize,2)-1
            gradient_w{layer} = stack_a{1,layer-1}*delta{1,layer}'/size(delta{1,layer},2);
            gradient_b{layer} = mean(delta{1,layer},2)/size(delta{1},2);
        end
        
        for layer = 1:size(netsize,2)-1
            % weight & bias update
            weights{1,layer} = weights{1,layer} - eta*gradient_w{1,layer};
            biases{1,layer} = biases{1,layer} - eta*gradient_b{1,layer};
        end
        
    end
    
    fprintf('Epoch %d / %d completed\n', epochs, n_epochs)
    test_results_mlp(test_x, test_y, weights, biases);
    clear test_result
end

%% (3) test results
fprintf('Test results:')
clear test_results accuracy
[test_results, accuracy] = test_results_mlp(test_x, test_y, weights, biases);

reports = ['\naccuracy ' num2str(accuracy) ...
    ' batchsize ' num2str(mini_batch_size) ...
    ' eta ' num2str(eta) ...
    ' epochs ' num2str(n_epochs) ...
    ' hidden_neurons ' num2str(n_hidden) ...
    ' cost_function ' cost_function];
fid = fopen('hidden_report.txt','a');
fprintf(fid, reports);
fclose(fid);
clear test_results

end
end