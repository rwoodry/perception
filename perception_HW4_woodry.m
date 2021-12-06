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

figure(1); subplot(3, 1, 1);

bar(ts, x);
ylim([0 1.5]);
title("Impulse Stimulus");
ylabel("Impulse Strength");

subplot(3, 1, 2);
plot(ys); hold on; plot(ys2); ylim([0 0.15]);
legend('Tau = 15', 'Tau = 45'); title("Impulse Responses (Impulse at T = 100ms)");
xlabel("Time Course (ms)"); ylabel("Response"); 

hold off;

% Plot the corresponding exponentials
subplot(3, 1, 3);
e1 = [zeros(1, 99), exp(-ts/tau)];
e2 = [zeros(1, 99), exp(-ts/tau2)];

plot(e1); hold on; plot(e2); hold off; ylim([0 1.5]);
title("Exponential Functions"); legend("Tau = 15", "Tau = 45");
xlabel("Time Course (ms)"); ylabel("Response"); 

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
step1(500:length(step1)) = 0;
step2(500:length(step2)) = 0;

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

figure(2); subplot(3, 1, 1);

bar(ts, x);
ylim([0 1.5]);
title("Step Stimulus");
ylabel("Step Strength");


subplot(3, 1, 2);
plot(ys1); hold on; plot(ys2);
legend('Tau = 15', 'Tau = 45'); title("Step Responses");
 ylabel("Response"); ylim([0 1.5]);
hold off;

subplot(3, 1, 3); 
plot(step1); hold on; plot(step2); ylim([0 1.5]);
legend('Tau = 15', 'Tau = 45'); title("Exponential Functions");
xlabel("Time Course (ms)"); ylabel("Response"); xlim([0 1000]);
hold off;

%% 1C)
% Compute and plot the responses to sinusoidal inputs
% y = h(w)x(w) = x(w) / (1 + d(w))

W = [10:10:500];
computed_amps = zeros(size(W));

figure(3);

for i=1:length(W)
    w = W(i);
    x_hat = zeros(size(ts));
    d_hat = zeros(size(ts));

    x_hat(w) = 1;
    d_hat(w) = 2*pi*w;
    y_hat = x_hat ./ (1 + d_hat);
    
    computed_amps(i) = max(abs(y_hat));
end

computed_phase = 0 + pi/2;
yyaxis left
bar(W, computed_amps);
ylabel("Computed Amplitude");
yyaxis right
yline(computed_phase, color = 'red', value = computed_phase);
ylabel("Computed Phase");
title("Predicted Amplitudes of Sinusoids convolved with the IIR Filter");
xlabel("Frequency");


%%
% 2
deltaT = 1;
tau = 25;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;
y5 = 0;
y6 = 0;
y7 = 0;

ys1 = zeros(size(ts));
ys2 = zeros(size(ts));
ys3 = zeros(size(ts));
ys4 = zeros(size(ts));
ys5 = zeros(size(ts));
ys6 = zeros(size(ts));
ys7 = zeros(size(ts));

f1 = zeros(size(ts));
f2 = zeros(size(ts));

x = zeros(size(ts));
x(100) = 1;

for t = 1:length(ts)
    deltaY1 = (deltaT / tau) * (-y1 + x(t)); y1 = y1 + deltaY1;
    ys1(t) = y1;
    
    deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
    ys2(t) = y2;
    
    deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
    ys3(t) = y3;
    
    deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
    ys4(t) = y4;
    
    deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
    ys5(t) = y5;

    deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
    ys6(t) = y6;

    deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;
    ys7(t) = y7;

    f1(t) = y3 - y7; f2(t) = y5 - y7;

end


figure(7);
plot(f1); hold on; plot(f2);hold off;


%% 3)
deltaT = 1;        % 1
deltaX = 1/120;     % 1/120
x = [-2:deltaX:2];
y = [-2:deltaX:2];
t = [0:deltaT:1000];

sigma = 0.1;
sf = 4;
evenFilt = exp(-(x.^2)./(2*sigma^2)) .* cos(2*pi*sf*x);
oddFilt = exp(-(x.^2)./(2*sigma^2)) .* sin(2*pi*sf*x);
integral = sum(evenFilt.^2 + oddFilt.^2);
evenFilt = evenFilt / integral;
oddFilt = oddFilt / integral;

% Create stimulus with impulse at mid x, mid y, t = 1
stim = zeros(length(x), length(y), length(t));
stim(round(length(x) / 2), round(length(x) / 2), 1) = 1;



tau = 25;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;
y5 = 0;
y6 = 0;
y7 = 0;

ys1 = zeros(size(stim));
ys2 = zeros(size(stim));
ys3 = zeros(size(stim));
ys4 = zeros(size(stim));
ys5 = zeros(size(stim));
ys6 = zeros(size(stim));
ys7 = zeros(size(stim));

f1_even = zeros(size(stim));
f1_odd = zeros(size(stim));
f1_even_t = zeros(size(stim));
f1_odd_t = zeros(size(stim));

f2_even = zeros(size(stim));
f2_odd = zeros(size(stim));
f2_even_t = zeros(size(stim));
f2_odd_t = zeros(size(stim));

f1 = zeros(size(stim));
f2 = zeros(size(stim));


for tt = 1:length(t)
    deltaY1 = (deltaT / tau) * (-y1 + stim(:, :, tt)); y1 = y1 + deltaY1;
    ys1(:, :, tt) = y1;
    
    deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
    ys2(:, :, tt) = y2;
    
    deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
    ys3(:, :, tt) = y3;
    
    deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
    ys4(:, :, tt) = y4;
    
    deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
    ys5(:, :, tt) = y5;

    deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
    ys6(:, :, tt) = y6;

    deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;
    ys7(:, :, tt) = y7;

    f1(:, :, tt) = y3 - y7; 
    f2(:, :, tt) = y5 - y7;

    f1_even(:, :, tt) = conv2(f1(:, :, tt), evenFilt, 'same');
    f1_odd(:, :, tt) = conv2(f1(:, :, tt), oddFilt, 'same');
    f1_even_t(:, :, tt) = conv2(f1(:, :, tt), evenFilt', 'same');
    f1_odd_t(:, :, tt) = conv2(f1(:, :, tt), oddFilt', 'same');
    
    f2_even(:, :, tt) = conv2(f2(:, :, tt), evenFilt, 'same');
    f2_odd(:, :, tt) = conv2(f2(:, :, tt), oddFilt, 'same');
    f2_even_t(:, :, tt) = conv2(f2(:, :, tt), evenFilt', 'same');
    f2_odd_t(:, :, tt) = conv2(f2(:, :, tt), oddFilt', 'same');

end

%% 3a)
figure(8);

subplot(2, 2, 3);
imagesc(squeeze(f1_even(round(length(x)/2), :, :))');
title("Even Fast"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(2, 2, 1);
imagesc(squeeze(f1_odd(round(length(x)/2),:, :))');
title("Odd Fast"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(2, 2, 4);
imagesc(squeeze(f2_even(round(length(x)/2), :, :))');
title("Even Slow"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(2, 2, 2);
imagesc(squeeze(f2_odd(round(length(x)/2), :, :))');
title("Odd Slow"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");


%% 3b)
leftEven = f1_odd + f2_even;
leftOdd = -f2_odd + f1_even;
rightEven = -f1_odd + f2_even;
rightOdd = f2_odd + f1_even;

figure(9);

subplot(2, 2, 1);
imagesc(squeeze(leftEven(round(length(x)/2), :, :))');
title("Left Even"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(2, 2, 2);
imagesc(squeeze(leftOdd(round(length(x)/2),:, :))');
title("Left Odd"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(2, 2, 3);
imagesc(squeeze(rightEven(round(length(x)/2), :, :))');
title("Right Even"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(2, 2, 4);
imagesc(squeeze(rightOdd(round(length(x)/2), :, :))');
title("Right Odd"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");


%% 3c)
leftEnergy = leftEven .^2 + leftOdd.^2;
rightEnergy = rightEven .^ 2 + rightOdd.^2;

figure(10);
subplot(1, 2, 1);
imagesc(squeeze(leftEnergy(round(length(x)/2), :, :))');
title("Left Energy"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");

subplot(1, 2, 2);
imagesc(squeeze(rightEnergy(round(length(x)/2), :, :))');
title("Right Energy"); xlabel("Visual angle (deg)"); ylabel("Time (ms)");


%% 3d)
deltaT = 1;        % 1
deltaX = 1/120;     % 1/120
x = [-2:deltaX:2];
y = [-2:deltaX:2];
t = [0:deltaT:1000];

sigma = 0.1;
sf = 4;
evenFilt = exp(-(x.^2)./(2*sigma^2)) .* cos(2*pi*sf*x);
oddFilt = exp(-(x.^2)./(2*sigma^2)) .* sin(2*pi*sf*x);
integral = sum(evenFilt.^2 + oddFilt.^2);
evenFilt = evenFilt / integral;
oddFilt = oddFilt / integral;



% Create stimulus with impulse at mid x, mid y, t = 1
stim = zeros(length(x), length(y), length(t));
stim(round(length(x) / 2), round(length(x) / 2), 1) = 1;
% Create 4 drifting sinusoid stimuli
Hz = 8;
cpd = sf;

drift_up = zeros(size(stim));
drift_down = zeros(size(stim));
drift_left = zeros(size(stim));
drift_right = zeros(size(stim));

units_per_degree = length(x)/ 4;
P = units_per_degree/cpd;
P_over_time = 1000/Hz;
delta_phase = 2*pi/P_over_time;

for tt=1:length(t)
    drift_up(:, :, tt) = mkSine([length(x) length(y)], P, pi/2, 1, ...
        delta_phase * tt);

    drift_right(:, :, tt) = mkSine([length(x) length(y)], P, pi, 1, ...
        delta_phase* tt);

    drift_left(:, :, tt) = mkSine([length(x) length(y)], P, 2*pi, 1, ...
        delta_phase*tt);

    drift_down(:, :, tt) = mkSine([length(x) length(y)], P, 3/2*pi, 1, ...
        delta_phase*tt);
end


%Uncomment below to play movie of the four stimuli

figure(11);

for tt = 1:length(t)
    subplot(2, 2, 1);
    imagesc(drift_up(:, :, tt));
    title("Up");

    subplot(2, 2, 2);
    imagesc(drift_down(:, :, tt));
    title("Down");

    subplot(2, 2, 3);
    imagesc(drift_left(:, :, tt));
    title("Left");

    subplot(2, 2, 4);
    imagesc(drift_right(:, :, tt));
    title("Right"); 
    
    pause(0.001);
end

%% 3d) Drift Left
tau = 25;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;
y5 = 0;
y6 = 0;
y7 = 0;

f1_even = zeros(size(stim));
f1_odd = zeros(size(stim));
f1_even_t = zeros(size(stim));
f1_odd_t = zeros(size(stim));

f2_even = zeros(size(stim));
f2_odd = zeros(size(stim));
f2_even_t = zeros(size(stim));
f2_odd_t = zeros(size(stim));

for tt = 1:length(t)
    deltaY1 = (deltaT / tau) * (-y1 + drift_left(:, :, tt)); y1 = y1 + deltaY1;
    deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
    deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
    deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
    deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
    deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
    deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;

    f1 = y3 - y7; 
    f2 = y5 - y7;

    f1_even(:, :, tt) = conv2(f1, evenFilt, 'same');
    f1_odd(:, :, tt) = conv2(f1, oddFilt, 'same');
    f1_even_t(:, :, tt) = conv2(f1, evenFilt', 'same');
    f1_odd_t(:, :, tt) = conv2(f1, oddFilt', 'same');
    
    f2_even(:, :, tt) = conv2(f2, evenFilt, 'same');
    f2_odd(:, :, tt) = conv2(f2, oddFilt, 'same');
    f2_even_t(:, :, tt) = conv2(f2, evenFilt', 'same');
    f2_odd_t(:, :, tt) = conv2(f2, oddFilt', 'same');
end

disp("Integration Complete");

leftEven = f1_odd + f2_even;
leftOdd = -f2_odd + f1_even;
rightEven = -f1_odd + f2_even;
rightOdd = f2_odd + f1_even;
upEven = f1_odd_t + f2_even_t;
upOdd = -f2_odd_t + f1_even_t;
downEven = -f1_odd_t + f2_even_t;
downOdd = f2_odd_t + f1_even_t;

leftEnergy = leftEven.^2 + leftOdd.^2;
rightEnergy = rightEven.^ 2 + rightOdd.^2;
upEnergy = upEven.^2 + upOdd.^2;
downEnergy = downEven.^ 2 + downOdd.^2;

disp("Energy Computed");

figure('NumberTitle', 'off', 'Name', 'Drift Left');
subplot(2, 2, 1);
plot(squeeze(leftEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(leftEven(round(length(x)/2), round(length(x)/2), :)));
title("Leftward Selective");
plot(squeeze(leftOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 2);
plot(squeeze(rightEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(rightEven(round(length(x)/2), round(length(x)/2), :)));
title("Rightward Selective");
plot(squeeze(rightOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 3);
plot(squeeze(downEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(downEven(round(length(x)/2), round(length(x)/2), :)));
title("Downward Selective");
plot(squeeze(downOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 4);
plot(squeeze(upEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(upEven(round(length(x)/2), round(length(x)/2), :)));
title("Upward Selective");
plot(squeeze(upOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;


%% 3d) Drift Right
tau = 25;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;
y5 = 0;
y6 = 0;
y7 = 0;

f1_even = zeros(size(stim));
f1_odd = zeros(size(stim));
f1_even_t = zeros(size(stim));
f1_odd_t = zeros(size(stim));

f2_even = zeros(size(stim));
f2_odd = zeros(size(stim));
f2_even_t = zeros(size(stim));
f2_odd_t = zeros(size(stim));

for tt = 1:length(t)
    deltaY1 = (deltaT / tau) * (-y1 + drift_right(:, :, tt)); y1 = y1 + deltaY1;
    deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
    deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
    deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
    deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
    deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
    deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;

    f1 = y3 - y7; 
    f2 = y5 - y7;

    f1_even(:, :, tt) = conv2(f1, evenFilt, 'same');
    f1_odd(:, :, tt) = conv2(f1, oddFilt, 'same');
    f1_even_t(:, :, tt) = conv2(f1, evenFilt', 'same');
    f1_odd_t(:, :, tt) = conv2(f1, oddFilt', 'same');
    
    f2_even(:, :, tt) = conv2(f2, evenFilt, 'same');
    f2_odd(:, :, tt) = conv2(f2, oddFilt, 'same');
    f2_even_t(:, :, tt) = conv2(f2, evenFilt', 'same');
    f2_odd_t(:, :, tt) = conv2(f2, oddFilt', 'same');
end

leftEven = f1_odd + f2_even;
leftOdd = -f2_odd + f1_even;
rightEven = -f1_odd + f2_even;
rightOdd = f2_odd + f1_even;
upEven = f1_odd_t + f2_even_t;
upOdd = -f2_odd_t + f1_even_t;
downEven = -f1_odd_t + f2_even_t;
downOdd = f2_odd_t + f1_even_t;

leftEnergy = leftEven.^2 + leftOdd.^2;
rightEnergy = rightEven.^ 2 + rightOdd.^2;
upEnergy = upEven.^2 + upOdd.^2;
downEnergy = downEven.^ 2 + downOdd.^2;


figure('NumberTitle', 'off', 'Name', 'Drift Right');
subplot(2, 2, 1);
plot(squeeze(leftEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(leftEven(round(length(x)/2), round(length(x)/2), :)));
title("Leftward Selective");
plot(squeeze(leftOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 2);
plot(squeeze(rightEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(rightEven(round(length(x)/2), round(length(x)/2), :)));
title("Rightward Selective");
plot(squeeze(rightOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 3);
plot(squeeze(downEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(downEven(round(length(x)/2), round(length(x)/2), :)));
title("Downward Selective");
plot(squeeze(downOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 4);
plot(squeeze(upEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(upEven(round(length(x)/2), round(length(x)/2), :)));
title("Upward Selective");
plot(squeeze(upOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;


%% 3d) Drift Up
tau = 25;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;
y5 = 0;
y6 = 0;
y7 = 0;

f1_even = zeros(size(stim));
f1_odd = zeros(size(stim));
f1_even_t = zeros(size(stim));
f1_odd_t = zeros(size(stim));

f2_even = zeros(size(stim));
f2_odd = zeros(size(stim));
f2_even_t = zeros(size(stim));
f2_odd_t = zeros(size(stim));

for tt = 1:length(t)
    deltaY1 = (deltaT / tau) * (-y1 + drift_up(:, :, tt)); y1 = y1 + deltaY1;
    deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
    deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
    deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
    deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
    deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
    deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;

    f1 = y3 - y7; 
    f2 = y5 - y7;

    f1_even(:, :, tt) = conv2(f1, evenFilt, 'same');
    f1_odd(:, :, tt) = conv2(f1, oddFilt, 'same');
    f1_even_t(:, :, tt) = conv2(f1, evenFilt', 'same');
    f1_odd_t(:, :, tt) = conv2(f1, oddFilt', 'same');
    
    f2_even(:, :, tt) = conv2(f2, evenFilt, 'same');
    f2_odd(:, :, tt) = conv2(f2, oddFilt, 'same');
    f2_even_t(:, :, tt) = conv2(f2, evenFilt', 'same');
    f2_odd_t(:, :, tt) = conv2(f2, oddFilt', 'same');
end

leftEven = f1_odd + f2_even;
leftOdd = -f2_odd + f1_even;
rightEven = -f1_odd + f2_even;
rightOdd = f2_odd + f1_even;
upEven = f1_odd_t + f2_even_t;
upOdd = -f2_odd_t + f1_even_t;
downEven = -f1_odd_t + f2_even_t;
downOdd = f2_odd_t + f1_even_t;

leftEnergy = leftEven.^2 + leftOdd.^2;
rightEnergy = rightEven.^ 2 + rightOdd.^2;
upEnergy = upEven.^2 + upOdd.^2;
downEnergy = downEven.^ 2 + downOdd.^2;


figure('NumberTitle', 'off', 'Name', 'Drift Up');
subplot(2, 2, 1);
plot(squeeze(leftEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(leftEven(round(length(x)/2), round(length(x)/2), :)));
title("Leftward Selective");
plot(squeeze(leftOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 2);
plot(squeeze(rightEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(rightEven(round(length(x)/2), round(length(x)/2), :)));
title("Rightward Selective");
plot(squeeze(rightOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 3);
plot(squeeze(downEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(downEven(round(length(x)/2), round(length(x)/2), :)));
title("Downward Selective");
plot(squeeze(downOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 4);
plot(squeeze(upEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(upEven(round(length(x)/2), round(length(x)/2), :)));
title("Upward Selective");
plot(squeeze(upOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
xlim([0 1000]); ylim([-.4 1.2]); hold off;


%% 3d) Drift Down
tau = 25;
y1 = 0;
y2 = 0;
y3 = 0;
y4 = 0;
y5 = 0;
y6 = 0;
y7 = 0;

f1_even = zeros(size(stim));
f1_odd = zeros(size(stim));
f1_even_t = zeros(size(stim));
f1_odd_t = zeros(size(stim));

f2_even = zeros(size(stim));
f2_odd = zeros(size(stim));
f2_even_t = zeros(size(stim));
f2_odd_t = zeros(size(stim));

for tt = 1:length(t)
    deltaY1 = (deltaT / tau) * (-y1 + drift_down(:, :, tt)); y1 = y1 + deltaY1;
    deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
    deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
    deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
    deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
    deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
    deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;

    f1 = y3 - y7; 
    f2 = y5 - y7;

    f1_even(:, :, tt) = conv2(f1, evenFilt, 'same');
    f1_odd(:, :, tt) = conv2(f1, oddFilt, 'same');
    f1_even_t(:, :, tt) = conv2(f1, evenFilt', 'same');
    f1_odd_t(:, :, tt) = conv2(f1, oddFilt', 'same');
    
    f2_even(:, :, tt) = conv2(f2, evenFilt, 'same');
    f2_odd(:, :, tt) = conv2(f2, oddFilt, 'same');
    f2_even_t(:, :, tt) = conv2(f2, evenFilt', 'same');
    f2_odd_t(:, :, tt) = conv2(f2, oddFilt', 'same');
end

leftEven = f1_odd + f2_even;
leftOdd = -f2_odd + f1_even;
rightEven = -f1_odd + f2_even;
rightOdd = f2_odd + f1_even;
upEven = f1_odd_t + f2_even_t;
upOdd = -f2_odd_t + f1_even_t;
downEven = -f1_odd_t + f2_even_t;
downOdd = f2_odd_t + f1_even_t;

leftEnergy = leftEven.^2 + leftOdd.^2;
rightEnergy = rightEven.^ 2 + rightOdd.^2;
upEnergy = upEven.^2 + upOdd.^2;
downEnergy = downEven.^ 2 + downOdd.^2;
%%
figure('NumberTitle', 'off', 'Name', 'Drift Down');
subplot(2, 2, 1);
plot(squeeze(leftEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(leftEven(round(length(x)/2), round(length(x)/2), :)));
title("Leftward Selective");
plot(squeeze(leftOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
 hold off; xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 2);
plot(squeeze(rightEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(rightEven(round(length(x)/2), round(length(x)/2), :)));
title("Rightward Selective");
plot(squeeze(rightOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
 hold off; xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 3);
plot(squeeze(downEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(downEven(round(length(x)/2), round(length(x)/2), :)));
title("Downward Selective");
plot(squeeze(downOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
hold off; xlim([0 1000]); ylim([-.4 1.2]); hold off;

subplot(2, 2, 4);
plot(squeeze(upEnergy(round(length(x)/2), round(length(x)/2), :)));
hold on;
plot(squeeze(upEven(round(length(x)/2), round(length(x)/2), :)));
title("Upward Selective");
plot(squeeze(upOdd(round(length(x)/2), round(length(x)/2), :)));
xlabel("Time (ms)"); ylabel("Response"); legend("Energy", "Even", "Odd");
hold off; xlim([0 1000]); ylim([-.4 1.2]); hold off;


%% 4a)
contrasts = [1 5 10 25 50 100];
sigma = 3;      % sigma value that yields a response of .40 at 10 % contrast, 
% which is half the maximum response: 80%.
upEnergy_byContrast = zeros(1, length(contrasts));
downEnergy_byContrast = zeros(1, length(contrasts));
rightEnergy_byContrast = zeros(1, length(contrasts));
leftEnergy_byContrast = zeros(1, length(contrasts));

for c=1:length(contrasts)
    contrast = contrasts(c);
    tau = 25;
    y1 = 0;
    y2 = 0;
    y3 = 0;
    y4 = 0;
    y5 = 0;
    y6 = 0;
    y7 = 0;
    
    f1_even = zeros(size(stim));
    f1_odd = zeros(size(stim));
    f1_even_t = zeros(size(stim));
    f1_odd_t = zeros(size(stim));
    
    f2_even = zeros(size(stim));
    f2_odd = zeros(size(stim));
    f2_even_t = zeros(size(stim));
    f2_odd_t = zeros(size(stim));

    stim = drift_right .* contrast;
    
    for tt = 1:length(t)
        deltaY1 = (deltaT / tau) * (-y1 + stim(:, :, tt)); y1 = y1 + deltaY1;
        deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
        deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
        deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
        deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
        deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
        deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;
    
        f1 = y3 - y7; 
        f2 = y5 - y7;
    
        f1_even(:, :, tt) = conv2(f1, evenFilt, 'same');
        f1_odd(:, :, tt) = conv2(f1, oddFilt, 'same');
        f1_even_t(:, :, tt) = conv2(f1, evenFilt', 'same');
        f1_odd_t(:, :, tt) = conv2(f1, oddFilt', 'same');
        
        f2_even(:, :, tt) = conv2(f2, evenFilt, 'same');
        f2_odd(:, :, tt) = conv2(f2, oddFilt, 'same');
        f2_even_t(:, :, tt) = conv2(f2, evenFilt', 'same');
        f2_odd_t(:, :, tt) = conv2(f2, oddFilt', 'same');
    end
    
    leftEven = f1_odd + f2_even;
    leftOdd = -f2_odd + f1_even;
    rightEven = -f1_odd + f2_even;
    rightOdd = f2_odd + f1_even;
    upEven = f1_odd_t + f2_even_t;
    upOdd = -f2_odd_t + f1_even_t;
    downEven = -f1_odd_t + f2_even_t;
    downOdd = f2_odd_t + f1_even_t;
    
    leftEnergy = leftEven.^2 + leftOdd.^2;
    rightEnergy = rightEven.^ 2 + rightOdd.^2;
    upEnergy = upEven.^2 + upOdd.^2;
    downEnergy = downEven.^ 2 + downOdd.^2;

    sumEnergy = leftEnergy + rightEnergy + upEnergy + downEnergy;
    leftEnergyNorm = leftEnergy ./ (sumEnergy + sigma^2);
    rightEnergyNorm = rightEnergy ./ (sumEnergy + sigma^2);
    upEnergyNorm = upEnergy ./ (sumEnergy + sigma^2);
    downEnergyNorm = downEnergy ./ (sumEnergy + sigma^2);

    upEnergy_byContrast(:, c) = mean(upEnergyNorm(round(length(x)/2), round(length(x)/2), :));
    downEnergy_byContrast(:, c) = mean(downEnergyNorm(round(length(x)/2), round(length(x)/2), :));
    leftEnergy_byContrast(:, c) = mean(leftEnergyNorm(round(length(x)/2), round(length(x)/2), :));
    rightEnergy_byContrast(:, c) = mean(rightEnergyNorm(round(length(x)/2), round(length(x)/2), :));

end

%% Plotting
figure(15);
plot(contrasts, rightEnergy_byContrast);
hold on;
plot(contrasts, leftEnergy_byContrast);
plot(contrasts, upEnergy_byContrast);
plot(contrasts, downEnergy_byContrast);
set(gca, 'XScale', 'log')

hold off; title("Average energy response as a function of contrast");
legend("Right", "Left", "Up", "Down"); xlabel("Contrast (%)");
ylim([0 1]); ylabel("Avg. Energy Response")


%% 4b)
contrasts = [1 5 10 25 50];
sigma = 3;      % sigma value that yields a response of .40 at 10 % contrast, 
% which is half the maximum response: 80%.
upEnergy_byContrast_UpRight = zeros(1, length(contrasts));
downEnergy_byContrast_UpRight = zeros(1, length(contrasts));
rightEnergy_byContrast_UpRight = zeros(1, length(contrasts));
leftEnergy_byContrast_UpRight = zeros(1, length(contrasts));

upEnergy_byContrast_Right = upEnergy_byContrast(1:length(contrasts));
downEnergy_byContrast_Right = downEnergy_byContrast(1:length(contrasts));
rightEnergy_byContrast_Right = rightEnergy_byContrast(1:length(contrasts));
leftEnergy_byContrast_Right = leftEnergy_byContrast(1:length(contrasts));



%%

for c=1:length(contrasts)
    contrast = contrasts(c);
    tau = 25;
    y1 = 0;
    y2 = 0;
    y3 = 0;
    y4 = 0;
    y5 = 0;
    y6 = 0;
    y7 = 0;
    
    f1_even = zeros(size(stim));
    f1_odd = zeros(size(stim));
    f1_even_t = zeros(size(stim));
    f1_odd_t = zeros(size(stim));
    
    f2_even = zeros(size(stim));
    f2_odd = zeros(size(stim));
    f2_even_t = zeros(size(stim));
    f2_odd_t = zeros(size(stim));

    stim = (drift_right .* contrast) + (drift_up .* contrast);
    
    for tt = 1:length(t)
        deltaY1 = (deltaT / tau) * (-y1 + stim(:, :, tt)); y1 = y1 + deltaY1;
        deltaY2 = (deltaT / tau) * (-y2 + y1); y2 = y2 + deltaY2;
        deltaY3 = (deltaT / tau) * (-y3 + y2);y3 = y3 + deltaY3;
        deltaY4 = (deltaT / tau) * (-y4 + y3); y4 = y4 + deltaY4;
        deltaY5 = (deltaT / tau) * (-y5 + y4); y5 = y5 + deltaY5;
        deltaY6 = (deltaT / tau) * (-y6 + y5); y6 = y6 + deltaY6;
        deltaY7 = (deltaT / tau) * (-y7 + y6); y7 = y7 + deltaY7;
    
        f1 = y3 - y7; 
        f2 = y5 - y7;
    
        f1_even(:, :, tt) = conv2(f1, evenFilt, 'same');
        f1_odd(:, :, tt) = conv2(f1, oddFilt, 'same');
        f1_even_t(:, :, tt) = conv2(f1, evenFilt', 'same');
        f1_odd_t(:, :, tt) = conv2(f1, oddFilt', 'same');
        
        f2_even(:, :, tt) = conv2(f2, evenFilt, 'same');
        f2_odd(:, :, tt) = conv2(f2, oddFilt, 'same');
        f2_even_t(:, :, tt) = conv2(f2, evenFilt', 'same');
        f2_odd_t(:, :, tt) = conv2(f2, oddFilt', 'same');
    end
    
    leftEven = f1_odd + f2_even;
    leftOdd = -f2_odd + f1_even;
    rightEven = -f1_odd + f2_even;
    rightOdd = f2_odd + f1_even;
    upEven = f1_odd_t + f2_even_t;
    upOdd = -f2_odd_t + f1_even_t;
    downEven = -f1_odd_t + f2_even_t;
    downOdd = f2_odd_t + f1_even_t;
    
    leftEnergy = leftEven.^2 + leftOdd.^2;
    rightEnergy = rightEven.^ 2 + rightOdd.^2;
    upEnergy = upEven.^2 + upOdd.^2;
    downEnergy = downEven.^ 2 + downOdd.^2;

    sumEnergy = leftEnergy + rightEnergy + upEnergy + downEnergy;
    leftEnergyNorm = leftEnergy ./ (sumEnergy + sigma^2);
    rightEnergyNorm = rightEnergy ./ (sumEnergy + sigma^2);
    upEnergyNorm = upEnergy ./ (sumEnergy + sigma^2);
    downEnergyNorm = downEnergy ./ (sumEnergy + sigma^2);

    upEnergy_byContrast_UpRight(:, c) = mean(upEnergyNorm(round(length(x)/2), round(length(x)/2), :));
    downEnergy_byContrast_UpRight(:, c) = mean(downEnergyNorm(round(length(x)/2), round(length(x)/2), :));
    leftEnergy_byContrast_UpRight(:, c) = mean(leftEnergyNorm(round(length(x)/2), round(length(x)/2), :));
    rightEnergy_byContrast_UpRight(:, c) = mean(rightEnergyNorm(round(length(x)/2), round(length(x)/2), :));

end

%% Plotting
figure(16);
subplot(2, 2, 1);
plot(contrasts, rightEnergy_byContrast(1:length(contrasts)), '-o', 'Color', 'blue');
hold on; set(gca, 'XScale', 'log'); 
plot(contrasts, rightEnergy_byContrast_UpRight, ':x', 'Color', 'blue');
hold off; title("Rightwards Selective");
ylim([0 1]); ylabel("Avg. Energy Response");  
xlabel("Contrast (%)"); legend("Right", "Up + Right")

subplot(2, 2, 2);
plot(contrasts, leftEnergy_byContrast(1:length(contrasts)),'-o', 'Color', 'red');
hold on; set(gca, 'XScale', 'log'); 
plot(contrasts, leftEnergy_byContrast_UpRight, ':x', 'Color', 'red');
hold off; title("Leftwards Selective");
ylim([0 1]); ylabel("Avg. Energy Response");  
xlabel("Contrast (%)"); legend(["Right" "Up + Right"])

subplot(2, 2, 3);
plot(contrasts, upEnergy_byContrast(1:length(contrasts)), '-o', 'Color', 'magenta');
hold on; set(gca, 'XScale', 'log'); 
plot(contrasts, upEnergy_byContrast_UpRight, ':x', 'Color', 'magenta');
hold off; title("Upwards Selective");
ylim([0 1]); ylabel("Avg. Energy Response");  
xlabel("Contrast (%)"); legend("Right", "Up + Right")

subplot(2, 2, 4);
plot(contrasts, downEnergy_byContrast(1:length(contrasts)), '-o', 'Color', 'green');
hold on; set(gca, 'XScale', 'log'); 
plot(contrasts, downEnergy_byContrast_UpRight, ':x', 'Color', 'green');
hold off; title("Downwards Selective");
ylim([0 1]); ylabel("Avg. Energy Response");  
xlabel("Contrast (%)"); legend("Right", "Up + Right")
