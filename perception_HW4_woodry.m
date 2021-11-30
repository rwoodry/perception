% V1 and Direction Selectivity
% Robert Woodry
% Perception HW4

%% 1A)
% Recursive Temporal Filter
% Compute the impulse response at t = 100
deltaT = 1;         % Time constant, ms
duration = 1000;    % ms

% Initialize arrays
ts = [0:deltaT:duration-deltaT]; %#ok<NBRAK> 
x = zeros(size(ts));
x(100) = 1;

% Compute & plot the IRs for two different tau values. Plot exponentials as
% well
tau = 15;

ys = zeros(size(ts));
y1 = 0;

for t = 1:length(ts)
    deltaY1 = (deltaT/tau) * (-y1 + x(t));
    y1 = y1 + deltaY1;
    ys(t) = y1;
end

tau2 = 45;

ys2 = zeros(size(ts));
y1 = 0;

for t = 1:length(ts)
    deltaY1 = (deltaT/tau2) * (-y1 + x(t));
    y1 = y1 + deltaY1;
    ys2(t) = y1;
end

figure(1); subplot(2, 1, 1);
plot(ys); hold on; plot(ys2);
legend('Tau = 15', 'Tau = 45'); title("Impulse Responses")
hold off;

% Plot the corresponding exponentials
subplot(2, 1, 2);
e1 = [zeros(1, 99), exp(-ts/tau)];
e2 = [zeros(1, 99), exp(-ts/tau2)];

plot(e1); hold on; plot(e2); hold off;
title("Exponential Functions"); legend("Tau = 15", "Tau = 45");

%% 1B)
% Compute and plot the step response

deltaT = 1;         % Time constant, ms
duration = 1000;    % ms

% Initialize arrays
ts = [0:deltaT:duration-deltaT]; %#ok<NBRAK> 
x = zeros(size(ts));
x(100:500) = 1;

step1 = [zeros(1, 99), 1 - exp(-ts/tau)];
step2 = [zeros(1, 99), 1 - exp(-ts/tau2)];

% Compute & plot the SRs for two different tau values. Plot exponentials as
% well
tau1 = 15;

ys1 = zeros(size(ts));
y1 = 0;

for t = 1:length(ts)
    deltaY1 = (deltaT/tau1) * (-y1 + x(t));
    y1 = y1 + deltaY1;
    ys1(t) = y1;
end

tau2 = 45;

ys2 = zeros(size(ts));
y1 = 0;

for t = 1:length(ts)
    deltaY1 = (deltaT/tau2) * (-y1 + x(t));
    y1 = y1 + deltaY1;
    ys2(t) = y1;
end

figure(2); subplot(2, 1, 1);
plot(ys1); hold on; plot(ys2);
legend('Tau = 15', 'Tau = 45'); title("Step Responses")
hold off;

subplot(2, 1, 2); 
plot(step1); hold on; plot(step2);
legend('Tau = 15', 'Tau = 45'); title("Step Responses - Exponentials")
hold off;

%% 1C)
% Compute and plot the responses to sinusoidal inputs
% y = h(w)x(w) = x(w) / (1 + d(w))
tau1 = 45;

ys1 = zeros(size(ts));
ys2 = zeros(size(ts));
ys3 = zeros(size(ts));
y1 = 0;
y2 = 0;
y3 = 0;

freq1 = 10;
freq2 = 50;
freq3 = 100;
sinusoid1 = sin(2 * pi * freq1 * ts/1000);
sinusoid2 = sin(2 * pi * freq2 * ts/1000); 
sinusoid3 = sin(2 * pi * freq3 * ts/1000);

figure(3);
subplot(2, 1, 1);
plot(ts, sinusoid1);
title("Sinusoidal Inputs")
hold on;
plot(ts, sinusoid2);
plot(ts, sinusoid3);
hold off;
legend("f = 4", "f = 16", "f = 64");

for t = 1:length(ts)
    % Sinusoid 1
    deltaY1 = (deltaT/tau1) * (-y1 + sinusoid1(t));
    y1 = y1 + deltaY1;
    ys1(t) = y1;

    % Sinusoid 2
    deltaY2 = (deltaT/tau1) * (-y2 + sinusoid2(t));
    y2 = y2 + deltaY2;
    ys2(t) = y2;

    % Sinusoid 3
    deltaY3 = (deltaT/tau1) * (-y3 + sinusoid3(t));
    y3 = y3 + deltaY3;
    ys3(t) = y3;
end

subplot(2, 1, 2);
plot(ts, ys1);
title("Sinusoidal Outputs")
hold on;
plot(ts, ys2);
plot(ts, ys3);
hold off;
legend("f = 10", "f = 50", "f = 100");

% TODO: COMPUTE FREQUENCY RESPONSE OF IIR FILTER
figure(4)
subplot(2, 1, 1);
plot(abs(fft(sinusoid1)));
hold on;
plot(abs(fft(sinusoid2)));
plot(abs(fft(sinusoid3)));
hold off;

subplot(2, 1, 2);
plot(abs(fft(ys1)/1000));
hold on;
plot(abs(fft(ys2)/1000));
plot(abs(fft(ys3)/1000));
hold off;

figure(5);
subplot(2, 1, 1);
exp_amps1 = abs(fft(sinusoid1))/ (1 + tau1 * 2 * pi * freq1);
exp_amps2 = abs(fft(sinusoid2))/ (1 + tau1 * 2 * pi * freq2);
exp_amps3 = abs(fft(sinusoid3))/ (1 + tau1 * 2 * pi * freq3);
plot(exp_amps1); hold on; plot(exp_amps2); plot(exp_amps3);hold off;


subplot(2, 1, 2);
plot(ifft(exp_amps1)*1000); hold on; 
plot((ifft(exp_amps2))*1000);
plot(ifft(exp_amps3)*1000);
xlim([0 1000]);
hold off;


