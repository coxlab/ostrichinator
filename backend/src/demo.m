function demo(imgfile, netpath, allwgpu, iter, faststp, stpthrs, varargin)

%run('matlab/vl_setupnn.m');
%make ARCH=glnxa64 MATLABROOT=/usr/local/MATLAB/R2014b ENABLE_IMREADJPEG=y ENABLE_GPU=y CUDAROOT=/usr/local/cuda
%tic; mcc -m -R -nojvm -R -nodisplay -R -singleCompThread -v demo.m; toc; delete mccExcludedFiles.log; delete readme.txt;

% -------------------------------------------------------------------------

% imgfile: path to target image
% netpath: path to network mat files
% iter: max allowed iteration; set to 1 for recognition mode (saliency map as by-product)
% faststp: 1 for stopping after target class prob becomes highest; 0 for stopping after target class prob reaches stpthrs
% stpthrs: stopping th for target class prob (works only when faststp = 0)
% varargin (6 variables): {caffnet_enable, caffnet_targetclass}, {vggs_...}, {vgg19_...}

% example: %m=[1 1 1]; c=1; demo('matimg/img0.png','models','0','50','1','1.0',m(1),c,m(2),c,m(3),c);

netfile = {'imagenet-caffe-ref.mat', ...
           'imagenet-vgg-s.mat', ...
           'imagenet-vgg-verydeep-19.mat'};

try
    % PROCESS PARAMETERS
    [allwgpu, iter, faststp, stpthrs, varargin{:}] = cell2double(allwgpu, iter, faststp, stpthrs, varargin{:});
    
    % LOAD IMAGE (SHOULD BE PNG)
    img = double(imread(imgfile));
    switch size(img,3)
        case 1, img = repmat(img, [1 1 3]);
        case 3, % NOTHING
        otherwise, error('IMREAD ERROR');
    end
    
    % LOAD NETWORK(S)
    tic;
    net = {}; tc = {};
    for i = 1:2:numel(varargin)
        if (varargin{i} == 1)
            net(end+1) = {load(fullfile(netpath, netfile{(i-1)/2+1}), 'layers')};
            tc(end+1)  = {[(varargin{i+1}) 1000]};
            
            temp = load(fullfile(netpath, netfile{(i-1)/2+1}), 'classes');
            [~,tempidx] = sortrows(temp.classes.name');
            net{end}.layers{end-1}.filters = net{end}.layers{end-1}.filters(:,:,:,tempidx);
        end
    end
    toc;
    
    % RUN DEMO
    alg = @ostrichinator_lite;
    
    tic;
    try
        [xmin, xsal, flag, ~, ~, chist] = alg(net, img, tc, allwgpu, iter, faststp, stpthrs);
    catch ERR % RETRY CPU MODE IF GPU MODE ERROR
        if (allwgpu == 0), rethrow(ERR);
        else [xmin, xsal, flag, ~, ~, chist] = alg(net, img, tc, 0, iter, faststp, stpthrs); end
    end
    toc;
    
    % PROCESS RESULTS
    for i = 1:numel(xsal)
        xsal{i} = sum(abs(xsal{i}), 3);
        xsal{i} = xsal{i} - min(xsal{i}(:));
        xsal{i} = xsal{i} / max(max(xsal{i}(:)), eps);
    end
    xsal = cat(3, xsal{:});
    xsal = min(sum(xsal, 3), 1); 
    
    xdff = xmin - img;
    xdff = xdff - min(xdff(:));
    xdff = xdff / max(max(xdff(:)), eps);
    xdff = round(xdff * 255);
    
    chist = [chist{:}];
    
    % WRITE RESULTS
    [imgpath,imgname,~] = fileparts(imgfile);
    
    imwrite(uint8(xmin), fullfile(imgpath, [imgname '-out.png']));
    
    xsal = gray2ind(xsal, 256);
    if (exist('parula','file') > 0), cm = @parula; else cm = @gray; end
    imwrite(xsal, cm(256), fullfile(imgpath, [imgname '-sal.png']));
    
    imwrite(uint8(xdff), fullfile(imgpath, [imgname '-dff.png']));
    
    fprintf('%s\n', num2str(chist(:,1)'  ));
    fprintf('%s\n', num2str(chist(:,end)'));
catch ERR
    fprintf('%s\n', ERR.getReport);
    flag = -1;
end

% 1=SUCCESSFUL FINISH; 0:UNSUCCESSFUL FINISH; -1:ERROR
fprintf('%d\n', flag);
fprintf('DONE');

% -------------------------------------------------------------------------

    function varargout = cell2double(varargin)
        
        varargout = cell(size(varargin));
        
        for c = 1:numel(varargin)
            if isa(varargin{c}, 'char'), varargout{c} = str2double(varargin{c});
            else varargout{c} = double(varargin{c}); end
        end
    end

end

