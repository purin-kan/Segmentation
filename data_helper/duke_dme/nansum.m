function y = nansum(x, dim)
% Minimal drop-in replacement for the Statistics and Machine Learning
% Toolbox's nansum (not installed on this machine). Only the 2-arg form is
% needed by external/oct_preprocess's DME branch (Preprocess.m line 107).
if nargin < 2
    y = sum(x, 'omitnan');
else
    y = sum(x, dim, 'omitnan');
end
end
