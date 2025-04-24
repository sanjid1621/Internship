%% Test PID Controller for PQD Correction with FPGA Simulation
clear; clc; close all;

% Load PQD dataset
load('pqd_data.mat'); % Contains signals, labels, Fs
load('pqd_frames.mat'); % Contains frame-based data

% Create time vector
T = 1/Fs;
t = 0:T:(size(signals,2)-1)*T;

% Initialize PID controller
pidController = PQD_PID_Controller();

% Select test case (1=Nominal, 2=Sag, etc.)
testCase = 3; % Change this to test different PQD types (1-6)
V_input = signals(testCase,:);
V_corrected = zeros(size(V_input));

% Simulate FPGA processing (would be real hardware)
frame_size = 256;
num_frames = floor(length(V_input)/frame_size);

% Process each frame for classification
disturbance_types = repmat("Nominal", 1, num_frames);
for i = 1:num_frames
    % Extract current frame
    start_idx = (i-1)*frame_size + 1;
    end_idx = i*frame_size;
    current_frame = V_input(start_idx:end_idx);
    
    % Save frame for Python processing (simulate FPGA)
    writematrix(current_frame, 'current_frame.csv');
    
    % Call Python classifier (simulate FPGA DPU)
    system('python classify_frame.py');
    
    % Read classification result
    fileID = fopen('disturbance_type.txt', 'r');
    disturbance_types(i) = fscanf(fileID, '%s');
    fclose(fileID);
end

% Run PID correction for each sample
current_disturbance = "Nominal";
for i = 1:length(V_input)
    % Update disturbance type at frame boundaries
    frame_num = ceil(i/frame_size);
    if frame_num <= length(disturbance_types)
        current_disturbance = disturbance_types(frame_num);
    end
    
    % Tune PID based on disturbance type
    pidController.tunePID(current_disturbance);
    
    % Update controller and get corrected voltage
    [V_corrected(i), ~] = pidController.update(V_input(i));
end

% Plot results
figure;
subplot(3,1,1);
plot(t, V_input, 'b', t, V_corrected, 'r');
legend('Disturbed Voltage', 'Corrected Voltage');
title(['PID Correction: ' char(labels(testCase))]);
xlabel('Time (s)'); ylabel('Voltage (V)'); grid on;
xlim([0 0.1]);

subplot(3,1,2);
plot(t, V_corrected - V_input);
title('Voltage Adjustment');
xlabel('Time (s)'); ylabel('Correction (V)'); grid on;
xlim([0 0.1]);

subplot(3,1,3);
frame_times = (0:num_frames-1)*frame_size*T;
stem(frame_times, categorical(disturbance_types));
title('FPGA Classification Results');
xlabel('Time (s)'); ylabel('Disturbance Type');
xlim([0 0.1]);