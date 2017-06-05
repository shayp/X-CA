function images = getImagesFromPath(ImagesPath)
for i = 1:length(ImagesPath)
    images(i).data = imread(ImagesPath{i});
    images(i).data = double(images(i).data) / 255.0;
end
end