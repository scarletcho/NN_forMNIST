function [test_results, accuracy] = test_results_mlp(test_x, test_y, weights, biases, output_activation)

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% test classification results (accuracy) in multi-layer perceptron
for sample = 1:size(test_x,2)
    [test_a_sample, test_z_sample] = feedforward_mlp(test_x(:,sample), ...
        weights, biases, output_activation);
    
    test_a_sample = test_a_sample{1,end};
    test_z_sample = test_z_sample{1,end};
    
    test_predictions = find(test_a_sample==max(test_a_sample));
    % [NOTE] In cases where there is multiple maximum values found,
    %        this simply takes the first number in the array.
    
    test_results{1,sample} = [find(test_y(:,sample)==1), test_predictions(1)];
    test_results{2,sample} = arrayfun(@isequal, ...
        test_results{1,sample}(:,1), test_results{1,sample}(:,2));
    
end

accuracy = sum(cell2mat(test_results(2,:)))/size(test_x,2)*100;
fprintf('accuracy: %.4g %%\n', accuracy)
end
