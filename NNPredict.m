function [ result ] = NNPredict( X, y, model)
%predict the y value by model caculated
W = model.W;
L = length(W);
N = length(X);
% X = [ones(N, 1), X];
% for i = 1:layer_d
%     
% end
for l = 1:L
    if l == 1
        tmp_X = [ones(N, 1), X];
    end
    w = W{l};
    if l ~= L
        tmp_X = [ones(N, 1), tanh(tmp_X * w)];
    else
        tmp_X = tanh(tmp_X * w);
    end
    if l == L
        predict_y = sign(tmp_X);
    end
end

diff = predict_y ~= y;
error_count = sum(diff);
error_rate = error_count/N;

result.predict_y = predict_y;
result.error_count = error_count;
result.error_rate = error_rate;

end

