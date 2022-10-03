function mat = nestedcell2mat(cells)

% SPDX-License-Identifier: BSD-3-Clause
tmp = vertcat(cells{:});
mat = vertcat(tmp{:});
end