clear;
% change directory here to test on different datasets 
directory = 'E:/Umich/HFIDnet/polygon/';
files = dir(directory);
files = files(3:end);
N = numel(files);
shape_spec = [];
acc = 5;
for i = 1555:1:2000
    if files(i).name(1) ~= '.'
        tic
        img_path = strcat(directory,files(i).name);
        temp = strsplit(files(i).name,'_');
        name = str2double(temp{1});
        gap = str2double(temp{2});
        temp = strsplit(temp{3},'.');
        thickness = str2double(temp{1});
        [TE,TM] = cal_spec(gap,thickness,acc,img_path);
        shape_spec(i,:) = [name,gap,thickness,TE,TM];
        disp(i);
        toc
    end
end

save 'shape_spec.mat' shape_spec