clear

load hospital;
X = [hospital.Age hospital.Weight];
X = [X, ones(size(X,1),1)];

Y = [20 120 1; 30 120 1; 40 168 1; 50 170 1; 60 171 1];   % New patients

Idx = knnsearch(X,Y,'K',3);

figure()
plot(X(:,1),X(:,2),'ko')
hold on
for i = 1:size(Idx,1)
    idx_pnt = Idx(i,:)';
    nn = X(idx_pnt,:);
    plot(Y(i,1),Y(i,2),'bd')
    plot(nn(:,1),nn(:,2),'r*')
end
hold off
axis equal
