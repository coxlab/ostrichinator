function demo(imgfile, netpath, allwgpu, iter, faststp, stpthrs, varargin)

% INPUT VARIABLES
% imgfile, path and filename to the target image, which must be PNG sized 227x227.
% netpath, path to the pretrained networks' mat files.
% allwgpu, enable GPU mode or not. 
% iter, maximally allowed running iterations; set to 1 for recognition only mode.
% faststp, 1 for fast stop (stopping after score of the target class becomes highest) and 0 for slow stop (stopping after probability of the target class reaches 'stpthrs').
% stpthrs, stopping threshold for probability of the target class when faststp is set to 0.
% varargin{1}, enable 'imagenet-caffe-ref' or not.
% varargin{2}, target class for 'imagenet-caffe-ref'.
% varargin{3}, enable 'imagenet-vgg-s' or not.
% varargin{4}, target class for 'imagenet-vgg-s'.
% varargin{5}, enable 'imagenet-vgg-verydeep-19' or not.
% varargin{6}, target class for 'imagenet-vgg-verydeep-19'.

% EXAMPLE: HACK TEST.PNG INTO CLASS 1 FOR ALL NETWORKS
% demo('static/test.png','backend/networks','0','50','1','1.0','1','1','1','1','1','1');

% EXAMPLE: COMPILE DEMO.M INTO EXECUTABLE
% mcc -m -R -nojvm -R -nodisplay -R -singleCompThread -v demo.m;

% -------------------------------------------------------------------------

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

% 1 = SUCCESSFUL FINISH; 0 = UNSUCCESSFUL FINISH; -1 = ERROR
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

