function [ NN ] = NNTrain( X, y, layer_d, T, alpha, w_range)
%basic neural network train implement
%   X:          train data X
%   y:          train data y
%   layer_d:    layer dimension
%   T:          sgd iteration count
%   alpha:      step rate
NN.layer_d = layer_d;
layer_count = length(layer_d);
NN.layer_count = layer_count;
L = layer_count - 1;
NN.L = L;
%init W for every layer W_ij
W = cell(layer_count-1, 1);
for l = 1:layer_count-1
    l_d = layer_d(l)+1;
    l_next_d = layer_d(l+1);
    W{l} = 2 * w_range * rand(l_d, l_next_d) - w_range;
end

N = length(y);

for i = 1:T
    n = randi(N,1,1);
    layer_X = cell(L,1);
    layer_S = cell(L,1);
    layer_delta = cell(L,1);
    layer_X{1} = [1,X(n,:)];
    %caculate layer_S & layer_X
    %forward
    for l = 1:L
        w = W{l};
        layer_S{l} = layer_X{l} * w;
        if l ~= L
            layer_X{l+1} = [1,tanh(layer_S{l})];
        end
    end
    %caculate delta
    %backward
    for l = L:-1:1
        if l == L
            layer_delta{l} = -2*(y(n) - tanh(layer_S{L}))/((exp(layer_S{L})+exp(-layer_S{L}))^2);
        else
            %layer_delta{l} = 2*(y(n) - layer_S{L});
            J = size(layer_S{l}, 2);
            K = size(layer_S{l+1}, 2);
            layer_delta{l} = zeros(1, J);
            for j = 1:J
                delta = 0;
                for k = 1:K
                    delta = delta + layer_delta{l+1}(1, k)*W{l+1}(j, k)*1/(exp(layer_S{l}(1,j))+exp(-layer_S{l}(1,j)))^2;
                end
                layer_delta{l}(1,j) = delta;
            end
        end
    end
    %caculate W_ij
    for l=1:L
        [I, J] = size(W{l});
        for i = 1:I
            for j = 1:J
                W{l}(i,j) = W{l}(i,j) - alpha*layer_X{l}(1, i)*layer_delta{l}(1,j);
            end
        end
    end
end

NN.W = W;
NN.T = T;

end

