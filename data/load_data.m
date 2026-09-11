%% MATLAB Data Loader & Replotter for F-NLFM Radar Datasets
% Loads CSV and NPZ data files and plots benchmark responses.

clear; clc; close all;

%% 1. Load 5.8 GHz Pulse Compression Profiles
if exist('dataset_1_pulse_compression_profiles.csv', 'file')
    data1 = readtable('dataset_1_pulse_compression_profiles.csv');
    figure('Color', 'w', 'Position', [100, 100, 900, 500]);
    plot(data1.delay_us, data1.lfm_db, 'b-', 'LineWidth', 1.2, 'DisplayName', 'LFM'); hold on;
    plot(data1.delay_us, data1.tangent_db, 'm-.', 'LineWidth', 1.2, 'DisplayName', 'Tangent NLFM');
    plot(data1.delay_us, data1.kaiser_posp_db, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Kaiser-POSP');
    plot(data1.delay_us, data1.fosm_db, 'r-', 'LineWidth', 2.0, 'DisplayName', 'Proposed FOSM');
    grid on; xlim([-2, 2]); ylim([-65, 5]);
    xlabel('Delay \tau (\mus)'); ylabel('Normalized Amplitude (dB)');
    title('Pulse Compression Benchmark at 5.8 GHz');
    legend('Location', 'northeast');
end

%% 2. Load Doppler Loss Curves
if exist('dataset_6_doppler_loss_curves.csv', 'file')
    data6 = readtable('dataset_6_doppler_loss_curves.csv');
    figure('Color', 'w', 'Position', [150, 150, 850, 480]);
    plot(data6.fd_khz, data6.loss_lfm_db, 'b--', 'LineWidth', 2.0, 'DisplayName', 'LFM'); hold on;
    plot(data6.fd_khz, data6.loss_tangent_db, 'm-.', 'LineWidth', 2.0, 'DisplayName', 'Tangent NLFM');
    plot(data6.fd_khz, data6.loss_fosm_db, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Proposed FOSM');
    grid on; xlim([0, 60]); ylim([0, 6]);
    xlabel('Doppler Frequency f_d (kHz)'); ylabel('Peak Doppler Mismatch Loss (dB)');
    title('Doppler Mismatch Loss vs. Radial Velocity Shift');
    legend('Location', 'northwest');
end
