function [blockMatrix, blockRhs, blockLhs] = assembleBlocks(runId, numBlock)
%VERIFYBLOCKASSEMBLY

prefix = strcat('run_', num2str(runId), '_');

blockRhs = cell(numBlock, 1);
for j = 1 : numBlock
    fileName = strcat(prefix, 'block_rhs_', num2str(j - 1), '.dat');
    blockRhs{j} = readmatrix(fileName);
end

blockLhs = cell(numBlock, 1);
for i = 1 : numBlock
    fileName = strcat(prefix, 'block_lhs_', num2str(i - 1), '.dat');
    blockLhs{i} = readmatrix(fileName);
end

blockMatrix = cell(numBlock, numBlock);
for i = 1 : numBlock
    m = size(blockRhs{i}, 1);
    for j = 1 : numBlock
        fileName = strcat(prefix, 'block_matrix_', num2str(i - 1), '_', ...
                          num2str(j - 1), '.dat');
        n = size(blockLhs{j}, 1);
        blockMatrix{i, j} = reshape(readmatrix(fileName), [m, n]);
    end
end

blockRhs    = cell2mat(blockRhs);
blockLhs    = cell2mat(blockLhs);
blockMatrix = cell2mat(blockMatrix);

solution = blockMatrix \ blockRhs;
abserror = norm(solution - blockLhs);
relerror = abserror / norm(blockLhs);

fprintf('-- absolute error = %e\n-- relative error = %e\n', ...
        abserror, relerror);

end

function matrix = readMatrixMarket(fileName)

data = readmatrix(fileName);
matrix = sparse(data(:, 1), data(:, 2), data(:, 3));

end