clf

n=20;
% addpath cmap;

dat = readmatrix("concentration.csv");
X = readmatrix("X.csv");
Y = readmatrix("Y.csv");
Z = readmatrix("Z.csv");


[t, n_pts] = size(dat);

for i=1:t

    c = dat(i, :);

    s = ones(n,n,n);
    color = flipud(ember);

    alpha = c;
    ind = c<1;
    alpha(ind) = 0;
    ind = c>=1;
    alpha(ind) = 1;
    nnz(alpha)

    scatter3(X,Y,Z,[],c, "filled", "AlphaData",alpha, "MarkerFaceAlpha","flat", ...
        "MarkerEdgeAlpha","flat", "AlphaDataMode", "manual")
    % colormap(flipud(fgreen));
    colormap(sky)
    title("t=" + num2str(i))
    view(20, 35)
    xlabel("x")
    ylabel("y")
    zlabel("z")
    colorbar
    drawnow
    pause(0.1)
end