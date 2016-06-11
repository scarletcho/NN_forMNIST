function [stack_a, stack_z] = feedforward_mlp(input, weights, biases, output_activation)

% 2016-06-05
% Yejin Cho (scarletcho@korea.ac.kr)

%% feedforward using sigmoid function in multi-layer perceptron
a_prev = input;
switch output_activation
    case 'softmax'
        for layer = 1:size(weights,2)-1
            [a_next, z_next] = feedforward(a_prev, ...
                weights{layer}, biases{layer}, 'sigmoid');
            stack_a{1,layer} = a_next;
            stack_z{1,layer} = z_next;
            a_prev = a_next;
        end
        
        [a_next, z_next] = feedforward(a_prev, ...
            weights{size(weights,2)}, biases{size(weights,2)}, 'softmax');
        stack_a{1,size(weights,2)} = a_next;
        stack_z{1,size(weights,2)} = z_next;
        
    otherwise
        for layer = 1:size(weights,2)
            [a_next, z_next] = feedforward(a_prev, weights{layer}, biases{layer}, 'sigmoid');
            stack_a{1,layer} = a_next;
            stack_z{1,layer} = z_next;
            a_prev = a_next;
        end
end
end
