n=100; % number of data points
x=ones(n,1)+rand(n,1)/2; % original data
m=20 % width of the filter
b=ones(m, 1)/m % filter coefficients
y = filter(b,1,x) % filtered data
t = 1:length(x); % “time” variable for plot
plot(t,x,t,y) % plot
legend('Original Data','Filtered Data') % label the graphs