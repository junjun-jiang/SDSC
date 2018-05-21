function Y = Reshape3D(X,BlockSize)

patch_size = BlockSize(2)+1-BlockSize(1);
X1 = [];
tX = X(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4),:);
for i = 1:size(tX,3)       
    X1 = [X1 reshape(tX(:,:,i),patch_size*patch_size,1)];                
end
Y = double(X1); 