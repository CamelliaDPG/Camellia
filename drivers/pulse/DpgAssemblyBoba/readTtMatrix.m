function ttMatrix = readTtMatrix(fileName)
%READTTVECTOR Read TT matrix from file generated using Boba savetxt
%
%   ttMatrix = readTtMatrix(fileName) reads a tt_matrix from file. The file
%   is assumed to have been created by running savetxt() on a Boba tensor
%   train matrix. The 'TT-Toolbox' package must already be loaded.

fileId = fopen(fileName);

% name
line = fgetl(fileId);

% dimension
line = fgetl(fileId);
words = split(line);
if length(words) ~= 2 || ~strcmp(words{1}, 'dimension')
    error('invalid format');
end
dimension = str2double(words{2});

% rows
line = fgetl(fileId);
words = split(line);
if length(words) ~= dimension + 1 || ~strcmp(words{1}, 'rows')
    error('invalid format');
end
rows = str2double(words(2:end));

% cols
line = fgetl(fileId);
words = split(line);
if length(words) ~= dimension + 1 || ~strcmp(words{1}, 'cols')
    error('invalid format');
end
cols = str2double(words(2:end));

% ranks
line = fgetl(fileId);
words = split(line);
if length(words) ~= dimension + 2 || ~strcmp(words{1}, 'ranks')
    error('invalid format');
end
ranks = str2double(words(2:end));

% cores
sizes = rows .* cols;
cores = cell(dimension, 1);
for d = 1 : dimension
    line = fgetl(fileId);
    if ~strcmp(line, strcat('core_', num2str(d - 1)))
        error('invalid format')
    end

    coreSize = sizes(d) * ranks(d) * ranks(d + 1);
    data = textscan(fileId, '%f\n', coreSize);

    if d == 1
        cores{d} = reshape(data{1}, [sizes(d), ranks(d + 1)]);
    elseif d == dimension
        cores{d} = reshape(data{1}, [sizes(d), ranks(d)]);
    else
        cores{d} = reshape(data{1}, [sizes(d), ranks(d), ranks(d + 1)]);
    end
end

fclose(fileId);

ttVector = tt_tensor(cores);
ttMatrix = tt_matrix(ttVector, rows, cols);

end