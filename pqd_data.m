clc; clear; close all;

%% Parameters
Fs = 5000; % Sampling frequency
T = 1/Fs;
t = 0:T:1-T;
N = length(t);

%% Generate random voltage and current signals
V_nominal = 230 * sin(2 * pi * 50 * t) + 10 * randn(size(t));
I_nominal = 10 * sin(2 * pi * 50 * t) + randn(size(t));

%% Introduce random PQD defects
V_sag = V_nominal .* (t < 0.4 | t > 0.6) + (0.6 + 0.2*rand) * V_nominal .* (t >= 0.4 & t <= 0.6);
V_swell = V_nominal .* (t < 0.2 | t > 0.8) + (1.2 + 0.2*rand) * V_nominal .* (t >= 0.2 & t <= 0.8);
V_harmonics = V_nominal + (20 + 10*rand) * sin(2 * pi * (100 + 50*rand) * t);
V_transient = V_nominal;
idx = randi([2000, 3000], 1, 50);
V_transient(idx) = V_transient(idx) + (50 + 50*randn(size(idx)));
V_fluctuation = V_nominal .* (1 + 0.1 * randn(size(t)));

%% Store signals for classification
signals = [V_nominal; V_sag; V_swell; V_harmonics; V_transient; V_fluctuation];
labels = categorical(["Nominal", "Sag", "Swell", "Harmonics", "Transient", "Fluctuation"]);

%% Create frame-based dataset for CNN
frame_size = 256; % Match FPGA processing block size
num_frames = floor(N/frame_size);
features = zeros(6, num_frames, frame_size); % 6 disturbance types

for i = 1:6
    for j = 1:num_frames
        start_idx = (j-1)*frame_size + 1;
        end_idx = j*frame_size;
        features(i,j,:) = signals(i, start_idx:end_idx);
    end
end

%% Save data in formats
save('pqd_data.mat', 'signals', 'labels', 'Fs', '-v7.3');
save('pqd_frames.mat', 'features', 'labels', 'Fs', 'frame_size', '-v7.3');

%% Visualization (optional)
figure;
tiledlayout(3,2);
for i = 1:6
    nexttile;
    plot(t, signals(i,:));
    title(char(labels(i)));
end