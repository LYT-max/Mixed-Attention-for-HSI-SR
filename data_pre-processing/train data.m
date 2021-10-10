clear;close all;
%% settings
% folder = 'Train';
savepath = '旋转100tongdao4bei18_36.h5';
size_input =18;
size_label =36;
scale = 2;
stride = 15;
%% initialization
dataa = zeros(size_input, size_input, 100, 1,1);
label = zeros(size_label, size_label, 100, 1,1);
count = 0;
%% generate data

load('pavia_train.mat');

pavia_225_715=im2uint16(pavia_225_715);
%image_old = chikusei1(:,:,:);

%% 
tic 
    A = modcrop(pavia_225_715, scale);
    [hei,wid,l] = size(A);
    A1 = imresize((gaussian_down_sample(A,scale)), scale,'bicubic');
    B = imrotate(A,90);
    C = imrotate(A,180);
    D = imrotate(A,270);
    B1 = imresize((gaussian_down_sample(B,scale)), scale,'bicubic');
    C1 = imresize((gaussian_down_sample(C,scale)), scale,'bicubic');
    D1 = imresize((gaussian_down_sample(D,scale)), scale,'bicubic');
    
    for x = 1: stride :hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            subim_label = A(x : x+size_label-1, y :y+size_label-1,:); 
            %不旋转操作
            subim_input =  A1(x : x+size_label-1, y :y+size_label-1,:);
            count=count+1;
            dataa(:, :, :,1,count) = subim_input;
            label(:, :, :,1,count) = subim_label;
        end
    end
 clear    pavia_225_715  A  A1
    [hei,wid,l] = size(B);
     for x = 1: stride :hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            subim_label = B(x : x+size_label-1, y :y+size_label-1,:); 
            %90
            subim_input =  B1(x : x+size_label-1, y :y+size_label-1,:);
            count=count+1;
            dataa(:, :, :,1,count) = subim_input;
            label(:, :, :,1,count) = subim_label;
        end
     end
  clear     B  B1   
       [hei,wid,l] = size(C);
     for x = 1: stride :hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            subim_label = C(x : x+size_label-1, y :y+size_label-1,:); 
           %180
            subim_input =  C1(x : x+size_label-1, y :y+size_label-1,:);
            count=count+1;
            dataa(:, :, :,1,count) = subim_input;
            label(:, :, :,1,count) = subim_label;
        end
    end
  clear     C  C1   
       [hei,wid,l] = size(D);
     for x = 1: stride :hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            subim_label = D(x : x+size_label-1, y :y+size_label-1,:); 
            %270
            subim_input =  D1(x : x+size_label-1, y :y+size_label-1,:);
            count=count+1;
            dataa(:, :, :,1,count) = subim_input;
            label(:, :, :,1,count) = subim_label;
        end
     end
clear     D  D1       
   
figure, imshow(dataa(:, :,50,1,1005));
figure, imshow(label(:, :,50,1,1005));
order = randperm(count);
dataa = dataa(:, :, :, order);
label = label(:, :, :, order); 

%save pavia_train.mat  dataa label
%% writing to HDF5
chunksz = 30;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
   last_read=(batchno-1)*chunksz;
   batchdata = dataa(:,:,:,last_read+1:last_read+chunksz); 
   batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
   disp([batchno, floor(count/chunksz)])
   startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
   curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
   created_flag = true;
   totalct = curr_dat_sz(end);
end
toc
h5disp(savepath);
