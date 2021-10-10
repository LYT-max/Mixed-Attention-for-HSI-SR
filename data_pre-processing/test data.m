clear;close all;
%% settings
size_input = 144;
size_label = 144;
scale = 2;
stride =150;
%% initialization
dataa = zeros(size_input, size_input, 100, 1,1);
label = zeros(size_label, size_label, 100, 1,1);
count = 0;

%% generate data

 load('paviactest.mat');
 paviactest=paviactest(:,:,1:100);
%image_old = chikusei1(:,:,:);

%% 
tic 

    pavia100 = modcrop(paviactest, scale);
    [hei,wid,l] = size(pavia100);
    for x = 1: stride :hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            
            subim_label = pavia100(x : x+size_label-1, y :y+size_label-1,:); 
            subim_input= gaussian_down_sample(subim_label,scale);
            %subim_input =  im_input(x : x+size_label-1, y :y+size_label-1,:);
            subim_input=imresize(subim_input, scale,'bicubic');
            figure, imshow(subim_label(:,:,100));
            figure, imshow(subim_input(:,:,100))
            count=count+1;
            dataa(:, :, :,count) = subim_input;
            label(:, :, :,count) = subim_label;

        end
    end
%save
 save '100_4_144_144' dataa  label



