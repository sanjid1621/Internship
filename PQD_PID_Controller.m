classdef PQD_PID_Controller
    properties
        Kp = 0.5;       % Proportional gain
        Ki = 0.1;       % Integral gain
        Kd = 0.05;      % Derivative gain
        Ts = 1/5000;    % Sampling time (from Fs = 5000 Hz)
        prevError = 0;   % Previous error for derivative term
        integral = 0;    % Integral accumulator
        setpoint = 230;  % Nominal voltage (V)
        outputLimits = [200, 260]; % Min/max voltage limits
    end
    
    methods
        function [correctedVoltage, controlAction] = update(obj, measuredVoltage)
            % Calculate error
            error = obj.setpoint - measuredVoltage;
            
            % PID terms
            proportional = obj.Kp * error;
            obj.integral = obj.integral + obj.Ki * error * obj.Ts;
            derivative = obj.Kd * (error - obj.prevError) / obj.Ts;
            
            % Calculate control action
            controlAction = proportional + obj.integral + derivative;
            
            % Apply saturation limits (fixed this line)
            controlAction = max(min(controlAction, obj.outputLimits(2)-obj.setpoint), ...
                              obj.outputLimits(1)-obj.setpoint);
            
            % Apply correction
            correctedVoltage = measuredVoltage + controlAction;
            
            % Anti-windup: Clamp integral term if saturated
            if controlAction >= (obj.outputLimits(2)-obj.setpoint) || ...
               controlAction <= (obj.outputLimits(1)-obj.setpoint)
                obj.integral = obj.integral - obj.Ki * error * obj.Ts;
            end
            
            % Update previous error
            obj.prevError = error;
        end
        
        function tunePID(obj, disturbanceType)
            % Tune PID gains based on disturbance type
            switch disturbanceType
                case 'Sag'
                    obj.Kp = 0.8;
                    obj.Ki = 0.2;
                    obj.Kd = 0.1;
                case 'Swell'
                    obj.Kp = 0.6;
                    obj.Ki = 0.15;
                    obj.Kd = 0.08;
                case 'Harmonics'
                    obj.Kp = 0.4;
                    obj.Ki = 0.1;
                    obj.Kd = 0.05;
                case 'Transient'
                    obj.Kp = 1.0;
                    obj.Ki = 0.05;
                    obj.Kd = 0.2;
                case 'Fluctuation'
                    obj.Kp = 0.5;
                    obj.Ki = 0.1;
                    obj.Kd = 0.05;
                otherwise % Default/Nominal
                    obj.Kp = 0.3;
                    obj.Ki = 0.05;
                    obj.Kd = 0.01;
            end
        end
    end
end