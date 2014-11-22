function demo(imgfile, netpath, allwgpu, iter, faststp, stpthrs, varargin)

%run('matlab/vl_setupnn.m');
%make ENABLE_GPU=y ARCH=glnxa64 MATLABROOT=/usr/local/MATLAB/R2013a CUDAROOT=/usr/local/cuda ENABLE_IMREADJPEG=y
%tic; mcc -m -R -nojvm -R -nodisplay -R -singleCompThread -v demo.m; toc; delete mccExcludedFiles.log; delete readme.txt;

% -------------------------------------------------------------------------

% imgfile: path to target image
% netpath: path to network mat files
% iter: max allowed iteration; set to 1 for recognition mode (saliency map as by-product)
% faststp: 1 for stopping after target class prob becomes highest; 0 for stopping after target class prob reaches stpthrs
% stpthrs: stopping th for target class prob (works only when faststp = 0)
% varargin (6 variables): {caffnet_enable, caffnet_targetclass}, {vggs_...}, {vgg19_...}

% example: %m=[1 1 1]; c=1; demo('testimg/test.png','models','0','50','1','1.0',m(1),c,m(2),c,m(3),c);

netfile = {'imagenet-caffe-ref.mat', ...
           'imagenet-vgg-s.mat', ...
           'imagenet-vgg-verydeep-19.mat'};

try
    % PROCESS PARAMETERS
    [allwgpu, iter, faststp, stpthrs, varargin{:}] = cell2double(allwgpu, iter, faststp, stpthrs, varargin{:});
    
    % LOAD IMAGE
    img = double(imread(imgfile)); % SHOULD BE PNG
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
        end
    end
    toc;
    
    % RUN DEMO
    tic;
    try
        [xmin, xsal, flag, ~, ~, chist] = ostritchfier_2p(net, img, tc, allwgpu, iter, faststp, stpthrs);
    catch err % retry cpu mode if gpu mode returns error
        if (allwgpu == 0), rethrow(err);
        else [xmin, xsal, flag, ~, ~, chist] = ostritchfier_2p(net, img, tc, 0, iter, faststp, stpthrs); end
    end
    toc;
    
    % PROCESS RESULTS
    xsal = sum(cat(4, xsal{:}), 4);
    xsal = sum(abs(xsal), 3);
    xsal = xsal - min(xsal(:));
    xsal = xsal / max(xsal(:));
    
    chist = [chist{:}];
    
    xmin = gather(xmin);
    xsal = gather(xsal);
    chist = gather(chist);
    
    % WRITE RESULTS
    [imgpath,imgname,~] = fileparts(imgfile);
    
    imwrite(uint8(xmin), fullfile(imgpath, [imgname '-out.png']));
    
    xsal = gray2ind(xsal, 256);
    if (exist('parula','file') > 0), cm = @parula; else cm = @gray; end % 2014b colormap
    imwrite(xsal, cm(256), fullfile(imgpath, [imgname '-sal.png']));
    
    fprintf('%s\n', num2str(chist(:,1)'  )); % initial class label(s)
    fprintf('%s\n', num2str(chist(:,end)')); % final class label(s)
catch 
    flag = -1;
end

fprintf('%d\n', flag); % 1: successful finish; 0: unsuccessful finish; -1: error
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

