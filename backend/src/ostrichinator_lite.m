function [xmin, xsal, flag, xhist, fhist, chist] = ostrichinator_lite(net, img, tclass, allwgpu, maxiter, faststp, stpthrs)

% PARAMETERS

dispimg = 0;
erlystp = 1;
disthrs = 5;
bfgshis = 10;
enbalnc = 1;
fulhist = 0;

if dispimg, clf; subplot(2,2,1); image(img/255); end
if faststp, stpthrs = 1.0; end

stpcond = @(x)all(x >= stpthrs);
rndmult = 16;

% SHARED VARS

fevalnum = 0;
res = []; sal = [];
xhist = []; fhist = []; chist = [];

% SETUP NETWORKS & TARGET FUNCTION

for n = 1:numel(net)
    net{n}.layers(end) = []; 
    if allwgpu, net{n} = vl_simplenn_move(net{n}, 'gpu'); end
end

xshp = [227 227 3];
tfuno = @(x)vl_simplenn_opt(net, reshape(x,xshp), tclass, false); % original
if faststp && enbalnc, tfunn = @(x)vl_simplenn_opt(net, reshape(x,xshp), tclass, numel(net) > 1); % normalized
else tfunn = tfuno; end

img = img - 128;

% EARLY TERM.

tfuno(img);
[iprogress, iclass] = cal_progress(res, tclass); chist = [chist {iclass(:)}];
if fulhist, xhist = [xhist {img}]; fhist = [fhist {res}]; end

flag = stpcond(iprogress);
if (flag || (maxiter==1)), xmin = img + 128; xsal = sal; return; end

% RUN SOLVER

caldis = @(x)norm(x(:)-img(:))/sqrt(prod(xshp));

lb = -128; ub = 127;
clip = @(x)min(max(x,lb),ub);
xini = img; xini = clip(xini(:));

lb = repmat(lb, size(xini));
ub = repmat(ub, size(xini));

rng(0);
fstphse = 1;
rderfix = 0;
rdclean = 0;

while (fevalnum < maxiter)
    if ~rderfix
        if fstphse
            fprintf('-------------------- First Phase (OPT) --------------------\n');
            
            options.maxIter = maxiter - fevalnum; options.corrections = bfgshis;
            [xmin, ~, fevaltmp] = minConf_TMP(tfunn,xini,lb,ub,options,@plot_progress);
            fevalnum = fevalnum + fevaltmp;
            
            if stpcond(cal_progress(res, tclass))
                if rdclean, fstphse = 0; if (caldis(xmin)<disthrs), break; end
                else rderfix = 1; end
            else
                xini = img + rndmult*randn(xshp); xini = clip(xini(:));
            end
        else
            fprintf('-------------------- Second Phase (SHR) --------------------\n');
            pdis = norm(xmin - img(:)) / sqrt(2); % SHRINKING
            pfun = @(x)clip(img(:) + min(pdis,norm(x-img(:)))*(x-img(:))/max(norm(x-img(:)),eps));
            xini = pfun(xmin);
            
            options.maxIter = maxiter - fevalnum; options.corrections = bfgshis;
            [xmintmp, ~, fevaltmp] = minConf_PQN(tfunn,xini,pfun,options,@plot_progress);
            fevalnum = fevalnum + fevaltmp;
            
            if stpcond(cal_progress(res, tclass))
                if rdclean, xmin = xmintmp; if (caldis(xmin)<disthrs), break; end
                else rderfix = 1; end
            else
                fprintf('Cannot improve more\n'); break;
            end
        end
    else
        fprintf('-------------------- Fixing Rounding Issue --------------------\n');
        
        rderfix = 0;
        
        if fstphse, xini = xmin;
        else xini = xmintmp; end
        
        pdis = norm(xini - img(:)); % MAINTAINING
        pfun = @(x)clip(img(:) + min(pdis,norm(x-img(:)))*(x-img(:))/max(norm(x-img(:)),eps));
        
        options.maxIter = maxiter - fevalnum; options.corrections = bfgshis;
        [xmintmp, ~, fevaltmp] = minConf_PQN(tfuno,xini,pfun,options,@plot_progress);
        fevalnum = fevalnum + fevaltmp;
        
        if rdclean
            fstphse = 0;
            xmin = xmintmp;
        else
            if fstphse, xini = img + rndmult*randn(xshp); xini = clip(xini(:));
            else fprintf('Cannot fix rounding issue in 2nd phase\n'); break; end
        end
    end
    
    fprintf('-------------------- FunEvals: %s --------------------\n', num2str(fevalnum));
end

% FINAL CHECK & OUTPUT

xmin = clip(round(xmin));
fprintf('Final distortion: %s\n', num2str(caldis(xmin),'%.2e'));

tfuno(xmin);
[fprogress, fclass] = cal_progress(res, tclass); chist = [chist {fclass(:)}];
if fulhist, xhist = [xhist {xmin}]; fhist = [fhist {res}]; end
flag = stpcond(fprogress);

xmin = reshape(xmin + 128, xshp);
xsal = sal;

% -------------------------------------------------------------------------
    
    function y = ii(x,i), y = ((1:numel(x)) == i)'; end
    function y = mi(x), [~,y] = max(x(:)); end

    function [f, g] = vl_simplenn_opt(net, x, ch, nrml)

        if ~allwgpu, x = single(x); mcast = @(x)single(x);
        else x = gpuArray(single(x)); mcast = @(x)gpuArray(single(x)); end
        
        f = 0;
        if ~isempty(ch), g = zeros(size(x)); else g = []; end
        
        [res, sal] = deal(cell(size(net)));
        
        for i = 1:numel(net)
            if ~isempty(ch)
                fch = ch{i}(1);
                if ~nrml
                    tgt = @(x)x(fch); der = @(x)mcast(ii(x,fch));
                else
                    tgt = @(x)x(fch)-max(x(:)); der = @(x)mcast(ii(x,fch)-ii(x,mi(x)));
                end
            else
                fch = 1;
                tgt = @(x)x(fch); der = [];
            end
            
            res{i} = vl_simplenn_fast(net{i}, x, der);
            
            sal{i} = double(gather(res{i}(1).dzdx));
            res{i} = double(gather(res{i}(end).x(:)));
            
            f = f + tgt(res{i});
            g = g + sum(sal{i}, 4);
        end
        
        f = -f; 
        g = -g(:);
    end
    
    function [stop, dispstr] = plot_progress(x)
        
        if dispimg
            dff = reshape(x,xshp)-img; dff = dff-min(dff(:)); dff = dff/max(dff(:));
            
            subplot(2,2,2); image(dff);
            subplot(2,2,3); image((reshape(x,xshp)+128)/255);
            subplot(2,2,4); imagesc(sum(abs(sum(cat(4,sal{:}),4)),3));

            drawnow;
        end
        
        stop = false;
        rdclean = 0;
        
        [cprogress, cclass, pscores] = cal_progress(res, tclass); chist = [chist {cclass(:)}];        
        if fulhist, xhist = [xhist {x}]; fhist = [fhist {res}]; end
            
        % STOP CONDITIONING
        if erlystp
            if stpcond(cprogress)
                restmp = res;
                % CHECK RESULTS AFTER ROUNDING, NO BACKWARD PASS HERE
		vl_simplenn_opt(net, clip(round(reshape(x,xshp))), [], 0);
                fevalnum = fevalnum + 0.5;
                stop = stpcond(cal_progress(res, tclass));
                
                res = restmp;
                if stop, rdclean = 1; end
            end
        end
        
        dispstr = num2str(cprogress, '%.2e/');
        dispstr = ['Progress: ' strtrim(dispstr(1:end-1)) ' Distortion: ' num2str(caldis(x),'%.2e')];
        if erlystp && stpcond(cprogress) && ~rdclean, dispstr = [dispstr ' Rounding Issue!']; end
    end

    function [progress, cc, pscores] = cal_progress(scores, tc)
              
        [progress, cc, pscores] = deal(zeros(1, numel(scores)));
        
        for i = 1:numel(scores)
            [~,cc(i)] = max(scores{i});
            pscores(i) = scores{i}(tc{i}(1));
            
            if faststp
		scores{i} = exp(scores{i} - max(scores{i}));
                progress(i) = scores{i}(tc{i}(1));
            else
                % SOFTMAX
                scores{i} = exp(scores{i} - max(scores{i}));
                scores{i} = scores{i} ./ sum(scores{i});

                progress(i) = scores{i}(tc{i}(1));
            end
        end
    end

end

