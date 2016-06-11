%% ANN assignment3: Neural Network Building from scratch
% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)
clc;clear all;close all;
mini_batch_sizeset = 50;
cost_function = 'MSE';
% mini_batch_sizeset = 10*mini_batch_sizeset;
% mini_batch_sizeset = [1 mini_batch_sizeset];

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
for batchsize = 1:numel(mini_batch_sizeset)
%% Q1-(a). Creating a 2-layered neural network
% (a) Try creating a network with just two layers - an input and an output
%     layer, no hidden layer - with 784 and 10 neurons (with the sigmoid
%     activation function at the output neuron), respectively, to do the
%     MNIST classification task.
%---------------------------------------------------------------------------
% (i)  Derive the learning algorithms for the weights and biases of the network.
% (ii) Implement the network with backpropagation algorithm from scratch.
%      Train the network using stochastic gradient descent with mini-batch
%      sizes of 1, 10, 20, and 100. What classification accuracies can you
%      achieve? Please discuss the results such as the learning curves and
%      final classification accuracies.
%% (1) Initial parameter set-up
netsize = [size(train_x,1) size(train_y,1)];
biases = randn(netsize(2),1);            % initial biases
weights = randn(netsize(1),netsize(2));  % initial weights

%% (2) Stochastic Gradient Descent
mini_batch_size = mini_batch_sizeset(batchsize);    % parameter setting for SGD
eta = 3;
n_epochs = 30;

% The 1st for loop: epochs
for epochs = 1:n_epochs
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
        
        [weights, biases, gradient_w, gradient_b] = ...
            update_mini_batch(mini_x_singlebatch, ...
            mini_y_singlebatch, weights, biases, eta, cost_function);
    end
    
    fprintf('Epoch %d / %d completed\n', epochs, n_epochs)
    test_results(test_x, test_y, weights, biases);
    
end

%% (3) test results
fprintf('Test results:')
clear test_results accuracy
[test_results, accuracy] = test_results(test_x, test_y, weights, biases);

reports = ['\naccuracy ' num2str(accuracy) ...
    ' batchsize ' num2str(mini_batch_size) ...
    ' eta ' num2str(eta) ...
    ' epochs ' num2str(n_epochs)];
fid = fopen('nohidden_report.txt','a');
fprintf(fid, reports);
fclose(fid);
clear test_results
end
% << Accuracy report >>
% Classification accuracy (%) from different mini-batch sizes (= n)
% accuracy 57 batchsize 1 eta 3 epochs 30
% accuracy 83.83 batchsize 10 eta 3 epochs 30
% accuracy 83.89 batchsize 20 eta 3 epochs 30 <-- optimal!
% accuracy 83.49 batchsize 30 eta 3 epochs 30
% accuracy 75.74 batchsize 40 eta 3 epochs 30
% accuracy 74.87 batchsize 50 eta 3 epochs 30
% accuracy 65.89 batchsize 60 eta 3 epochs 30
% accuracy 56.21 batchsize 70 eta 3 epochs 30
% accuracy 64.81 batchsize 80 eta 3 epochs 30
% accuracy 55.62 batchsize 90 eta 3 epochs 30
% accuracy 48.71 batchsize 100 eta 3 epochs 30
