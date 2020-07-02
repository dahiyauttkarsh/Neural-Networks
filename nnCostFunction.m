function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

rolled_y=zeros(num_labels,size(y,1));
for a = 1:m
    for b = 1:num_labels
        if y(a,1) == b
            rolled_y(b,a) = rolled_y(b,a) + 1;
        end
    end
end

h=zeros(num_labels,size(y,1));

for i =1:m
 
    x=X(i,:);
    x=x.';
    x=[ones(1,1);x];
    a1=x(:);
    z2=Theta1*a1;
    a2=sigmoid(z2);
    a2=[ones(1,1);a2];
    z3=Theta2*a2;
    a3=sigmoid(z3);
    hx = a3;
    h(:,i) = h(:,i) + hx;
    
    d_3 = h(:,i)-rolled_y(:,i);
    %d_3(10*1)
    T2 = Theta2;
    T2(:,[1])=[];
    
    d_2 = transpose(T2)*d_3.*sigmoidGradient(z2);
    
    %d_2 = d_2(2:end);
    Theta1_grad = Theta1_grad + d_2*transpose(a1);
    
    Theta2_grad = Theta2_grad + d_3*transpose(a2);
    
end
Theta1_grad = Theta1_grad/m;
    Theta2_grad = Theta2_grad/m;
ty = rolled_y;
h = h;
j = -ty.*log(h) - (1-ty).*log(1-h);

size(j);


J = sum(j(:))/m;

t_theta1 = Theta1;
t_theta2 = Theta2;
t_theta1(:,[1])=[];
t_theta2(:,[1])=[];
t_theta1=t_theta1.^2;
t_theta2=t_theta2.^2;
reg = (sum(t_theta1(:)) + sum(t_theta2(:)))*lambda/(2*m);

J = J + reg;

init_t_g1 = Theta1_grad(:,1);
end_t_g1 = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta1_grad = [init_t_g1,end_t_g1];

init_t_g2 = Theta2_grad(:,1);
end_t_g2 = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
Theta2_grad = [init_t_g2,end_t_g2];





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
