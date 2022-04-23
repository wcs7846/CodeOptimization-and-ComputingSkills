%% Application of SUPPORT VECTOR REGRESSION to estimate a non linear function
% Least Squares Support Vector Regression method
clc;
clear all;
close all;
%% Input and Output definition
% Define the input range x.
x = [1:150]; 
x_lenght=length(x);

%Define the original function to be estimated
y_orig=sin(0.15*x)+0.05*exp(x/30);

%Define the original function affected by noise that will generate the
%samples
y_noise=y_orig+0.3*randn(1,x_lenght);

% Define the training data for x
xs = [1:0.1:150];
xs_lenght=length(xs);

% Define the training data for y
y_train=sin(0.15*xs)+0.05*exp(xs/30)+0.3*randn(1,xs_lenght);

% Create an empty vector to hold the approximate function values.
ys = zeros(size(x));

%% Start the Learning Algorithm, LEAST SQUARES-SVR
[ alfa, bias ] = lssvr_train( xs, y_train, 100, 0.01);
[ ys ] = lssvr_predict( x, xs, alfa, bias, 0.01);

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
