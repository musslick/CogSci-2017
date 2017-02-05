clc;

% parameters
k = 3;  % number of desired equidistant vectors
cosine_val = 0;

ref_vec = zeros(10, 1);
ref_vec(1:k) = 1;

m = size(ref_vec, 1);    % number of dimensions
d = 0.5; % cosine distance
x_norm = 1; % norm of vectors
referenceVector = ones(m, 1);   % reference vector for second constraint

% initialize vectors
X = zeros(m, k);

% compute initial points
X(1,1) = 1;  % first point
X(1,2) = -1;  % second point

% generate set of equidistant points
for i = 3:k
    
    % compute height for new point along new dimension
    l  = sqrt(sum((X(:,i-1)) .^ 2));  % distance between previous point to origin
    d  = sqrt(sum((X(:,i-1) - X(:,i-2)) .^ 2));
    syms x
    h_sym = solve(sqrt(x^2+l^2) == d, x); % distance between points (should be equidistant)
    h = double(h_sym(1));
    
    X(i,i) = h;    % compute distance along new dimensionnew point
    
    % compute mean of existing points
    X_mean = mean(X(:,1:i),2);
    
    % subtract mean of existing points to center them around origin
    X(:,1:i) = X(:,1:i)  - repmat(X_mean,1,i);
end

% make sure that equidistant point have the same angle
alpha = acos(cosine_val);
l  = sqrt(sum((X(:,1)) .^ 2));
h = l / tan(alpha);

offset_vec = zeros(m, 1);
offset_vec(k+1) = h;

X_new(:, 1:k) = X(:, 1:k) - repmat(offset_vec, 1, k);

% align equidistant points to reference vector via Gram-Schmidt process

U = circshift(eye(m, m), 1, 2);
U(:,1) = ref_vec/norm(ref_vec);


for i = 1:size(U,2)
    
    v = U(L,i-1);
    
    for j = i:size(U,2)
        
        u = U(:, j);
        u_new = u - dot(u, v) * v;
        
        U(:, j) = u_new;
        
    end
    
end

W = X * U;



