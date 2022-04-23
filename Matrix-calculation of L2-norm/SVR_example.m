%% Application of SUPPORT VECTOR REGRESSION to estimate a non linear function
% Least Squares Support Vector Regression method
clc;
clear all;
close all;
%% Input and Output definition
% Define the input range x.
x = [1:150]'; 
x_lenght=length(x);

%Define the original function to be estimated
y_orig=sin(0.15*x)+0.05*exp(x/30);

%Define the original function affected by noise that will generate the
%samples
y_noise=y_orig+0.3*randn(x_lenght,1);

% Define the training data for x
xs = [1:0.1:150]';
xs_lenght=length(xs);

% Define the training data for y
y_train=sin(0.15*xs)+0.05*exp(xs/30)+0.3*randn(xs_lenght,1);

% Create an empty vector to hold the approximate function values.
ys = zeros(size(x));

%% Start the Learning Algorithm, LEAST SQUARES-SVR

A=zeros(xs_lenght,xs_lenght); % initialize matrix A
C=100; %Parameter defined to avoid overfitting
g=0.01;%Radial Basis Functon learning parameter, is equal to 1/2sigma^2

%Radial Basis function
for i=1:xs_lenght
    for j=1:xs_lenght
        A(i,j)=exp(-g*(xs(i,1)-xs(j,1))^2);
        if i==j
            A(i,j)= A(i,j)+1/C;
        end
    end
end

O=[0, ones(1,xs_lenght);
        ones(xs_lenght,1), A];

b=zeros(xs_lenght+1,1);
c=zeros(xs_lenght+1,1);

 for f=1:xs_lenght
     c(f+1,1)=y_train(f,1);
 end 

b=inv(O)*c; %it contains LaGrange multipliers and bias

% For convenience, we separate the LaGrange Multipliers and the Bias term

%Bias term
bias=b(1,1);

%Define LaGrange Multipliers alfa
alfa=zeros(x_lenght,1); 

for i=1:xs_lenght
    alfa(i,1)=b(i+1,1); 
end

q=zeros(xs_lenght,1);% Storage variable

for j=1:x_lenght
   
 for i=1:xs_lenght
    
     q(i,1)=alfa(i,1).*exp(-g*(x(j,1)-xs(i,1))^2);
   
 end
     ys(j,1)=sum(q)+bias;
end


%% show the result
figure(1);
hold on; 

% Plot the original function as a blue line.
plot(x, y_orig,'color','blue');

% Plot the noisy data as red poinyts
plot(x, y_noise, '.','color','red');

% Plot the approximated function as a red line.
plot(x, ys,'color','red');

legend('Original', 'Noisy Samples', 'Approximated');
title('LS-SVR Regression');
