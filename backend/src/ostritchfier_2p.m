% TODO: adaptive W; move to GPU?

function [xmin, xsal, flag, xhist, fhist, chist] = ostritchfier_2p(net, img, tclass, allwgpu, maxiter, faststp, stpthrs)

% PARAMETERS

dispimg = 0;
erlystp = 1;
%faststp = 1;
%stpthrs = 0.5;

if dispimg, clf; subplot(2,2,1); image(img/255); end
if faststp, stpthrs = 1.0; end

stpcond = @(x)all(x >= stpthrs); % TH. HERE
rndmult = 16;

% SHARED VARS

fevalnum = 0; %nodiff = false;
res = []; sal = [];
xhist = []; fhist = []; chist = [];

% SETUP NETWORKS & TARGET FUNCTION

for n = 1:numel(net)
    net{n}.layers(end) = []; 
    if allwgpu, net{n} = vl_simplenn_move(net{n}, 'gpu'); end
end

xshp = [227 227 3];
tfun = @(x)vl_simplenn_opt(net, reshape(x,xshp), tclass);

img = img - 128;

% EARLY TERM.

[~, grad] = tfun(img); % fill in res/sal, fix gradient (1e-2) if needed
fprintf('Norm of Initial Grad: %d\n', norm(grad));
[iprogress, iclass] = cal_progress(res, tclass); chist = [chist {iclass(:)}];

flag = stpcond(iprogress);
if (flag || (maxiter==1)), xmin = img + 128; xsal = sal; return; end

% FIX IMAGE (FOR 1ST PASS)

lb = -128; ub = 127; clip = @(x)min(max(x,lb),ub);

nratio = 1; % - (numel(unique(img))-1)/255;
if (norm(grad) > 1e-2), rndmult = 0; end
rng(0); xini = img + nratio*rndmult*randn(xshp); xini = clip(xini(:));

% RUN SOLVER

while (fevalnum < maxiter)
    if (fevalnum == 0)
        lb = repmat(lb, size(xini));
        ub = repmat(ub, size(xini));
        
        options.maxIter = maxiter - fevalnum;
        [xmin, ~, fevaltmp] = minConf_TMP(tfun,xini,lb,ub,options,@plot_progress);
        fevalnum = fevalnum + fevaltmp;
        %break; % no 2nd pass
    else
        pdis = norm(xmin - img(:)) / sqrt(2);
        pfun = @(x)clip(img(:) + pdis*(x-img(:))/norm(x-img(:)));
        xini = pfun(xmin);
        
        options.maxIter = maxiter - fevalnum;
        [xmintmp, ~, fevaltmp] = minConf_PQN(tfun,xini,pfun,options,@plot_progress);
        fevalnum = fevalnum + fevaltmp;
        
        if stpcond(cal_progress(res, tclass)), xmin = xmintmp; end
    end
end

% FINAL CHECK & OUTPUT

xmin = clip(round(xmin));

tfun(xmin);
[fprogress, fclass] = cal_progress(res, tclass); chist = [chist {fclass(:)}];
flag = stpcond(fprogress);

xmin = reshape(xmin + 128, xshp);
xsal = sal;

% -------------------------------------------------------------------------

    function [f, g] = vl_simplenn_opt(net, x, ch)
        
        x = single(x);
        if allwgpu, x = gpuArray(x); end
        
        f = 0;
        if ~isempty(ch), g = zeros(size(x)); else g = []; end
        
        [res, sal] = deal(cell(size(net)));
        
        % OR PARFOR
        for i = 1:numel(net)
            if ~isempty(ch)
                dzdy = zeros([1 1 ch{i}(2)],'like',x); dzdy(ch{i}(1)) = 1; fch = ch{i}(1);
            else
                dzdy = []; fch = 1;
            end
            
            res{i} = vl_simplenn(net{i}, x, dzdy);
            f = f + double(gather(res{i}(end).x(fch)));
            g = g + double(gather(res{i}(1).dzdx));
        end
        
        f = -f; 
        g = -g(:);
        
        for i = 1:numel(net)
            sal{i} = gather(res{i}(1).dzdx);
            res{i} = gather(res{i}(end).x(:));
        end
    end
    
    function [stop, dispstr] = plot_progress(x)
        
        stop = false;
        x = reshape(x, xshp);
               
        if dispimg           
            dff = x-img; dff = dff-min(dff(:)); dff = dff/max(dff(:));
            
            subplot(2,2,2); image(dff);
            subplot(2,2,3); image((x+128)/255);
            subplot(2,2,4); imagesc(sum(abs(sum(cat(4,sal{:}),4)),3));

            drawnow;
        end
        
        [cprogress, cclass] = cal_progress(res, tclass); chist = [chist {cclass(:)}];        
            
        % STOP CONDITIONING
        if erlystp
            if stpcond(cprogress)                
                % making sure results are the same after rounding
                vl_simplenn_opt(net, clip(round(x)), []); % NO SAL
                fevalnum = fevalnum + 0.5;
                stop = stpcond(cal_progress(res, tclass));
            end
        end
        
        dispstr = num2str(cprogress, '%+.5f/');
        dispstr = ['Progress: ' strtrim(dispstr(1:end-1)) ' FunEvalsBefore: ' num2str(fevalnum)];
    end

    function [progress, cc] = cal_progress(scores, tc)
              
        [progress, cc] = deal(zeros(1, numel(scores)));
        
        for i = 1:numel(scores)
            [~,cc(i)] = max(scores{i});
            
            if faststp
                progress(i) = scores{i}(tc{i}(1)) / max(scores{i});
            else
                % SOFTMAX
                scores{i} = exp(scores{i} - max(scores{i}));
                scores{i} = scores{i} ./ sum(scores{i});

                progress(i) = scores{i}(tc{i}(1));
            end
        end
    end

end

